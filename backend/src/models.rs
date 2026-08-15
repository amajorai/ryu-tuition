//! Domain types for the Tuition spine — the wire contract the sidecar serves, the
//! store persists, and the companion UI renders from.
//!
//! Conventions, applied uniformly and deliberately:
//!
//! - **Ids are TEXT with a typed prefix** (`sub_`, `src_`, `skl_`, `itm_`, `att_`,
//!   `ses_`, `cnd_`) wrapping a UUIDv4. The prefix is not decoration: there are no
//!   foreign keys behind any of the cross-table references (see [`crate::store`]
//!   for why), so a mis-wired id is otherwise invisible until it silently matches
//!   nothing.
//! - **Timestamps are `i64` epoch MILLIS**, never RFC-3339 strings — the hot
//!   predicate of this whole app is `due_at <= now`, and lexicographic string
//!   comparison is the wrong tool for a calendar. The one exception is
//!   [`Subject::exam_date`], which is a calendar *date* the user typed and must not
//!   drift with a timezone: it stays the `YYYY-MM-DD` string they entered.
//! - **Booleans are `bool` on the wire, `INTEGER` 0/1 in SQLite.**
//! - **Every field name is snake_case on the wire**, matching every other Ryu
//!   sidecar and the hook's KV payload.
//!
//! Enum columns carry no SQL `CHECK` constraint. The Rust enum plus its `parse` IS
//! the guard: a value that fails to parse degrades to a documented default rather
//! than failing a whole list query, which is what keeps one corrupt row from
//! blanking the subject list. The single place that rule is *not* applied is
//! [`AnswerKey`] — an unreadable answer must never silently become a gradeable one,
//! so it decodes to [`AnswerKey::Malformed`] and [`AnswerKey::encode`] refuses to
//! write it back.

use serde::{Deserialize, Serialize};

// ── Time + id helpers ──────────────────────────────────────────────────────────

/// Now, as epoch millis. Every `created_at` / `at` / `due_at` in this app is
/// produced here so there is exactly one clock read to stub in tests.
pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// A fresh prefixed id. See the module docs for why the prefix exists.
pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

pub const ID_SUBJECT: &str = "sub_";
pub const ID_SOURCE: &str = "src_";
pub const ID_SKILL: &str = "skl_";
pub const ID_ITEM: &str = "itm_";
pub const ID_ATTEMPT: &str = "att_";
pub const ID_SESSION: &str = "ses_";
pub const ID_CANDIDATE: &str = "cnd_";

/// The IANA zone a subject falls back to when its `timezone` is unset or unknown.
pub const DEFAULT_TIMEZONE: &str = "UTC";

/// Resolve an IANA zone name, falling back to UTC.
///
/// Never fails: a subject whose timezone string was hand-edited to nonsense must
/// still schedule reviews, just in UTC.
pub fn resolve_tz(name: &str) -> chrono_tz::Tz {
    name.trim().parse().unwrap_or(chrono_tz::UTC)
}

/// Midnight of `at_ms`'s local day in `tz`, as epoch millis.
///
/// SM-2 intervals here are whole days measured from a day boundary, not
/// `now + n*86400s` — that is what stops a review due "tomorrow" from sliding an
/// hour later every time a session runs at 23:58.
///
/// DST is the reason this is a loop rather than one call: a handful of zones
/// (Chile, Cuba, Iran historically) spring forward *at* midnight, so local
/// 00:00:00 does not exist on that date and `from_local_datetime` returns nothing.
/// The first hour of the day that does exist is the correct day boundary there.
pub fn day_start_ms(tz: chrono_tz::Tz, at_ms: i64) -> i64 {
    use chrono::TimeZone;
    let Some(utc) = chrono::Utc.timestamp_millis_opt(at_ms).single() else {
        return at_ms;
    };
    let date = utc.with_timezone(&tz).date_naive();
    for hour in 0..4 {
        let Some(naive) = date.and_hms_opt(hour, 0, 0) else {
            continue;
        };
        if let Some(local) = tz.from_local_datetime(&naive).earliest() {
            return local.timestamp_millis();
        }
    }
    at_ms
}

/// The day boundary `days` whole days after the local day containing `at_ms`.
/// This is what an SM-2 `interval_days` turns into.
pub fn day_start_plus_days(tz: chrono_tz::Tz, at_ms: i64, days: u32) -> i64 {
    use chrono::TimeZone;
    let Some(utc) = chrono::Utc.timestamp_millis_opt(at_ms).single() else {
        return at_ms;
    };
    let Some(date) = utc
        .with_timezone(&tz)
        .date_naive()
        .checked_add_days(chrono::Days::new(u64::from(days)))
    else {
        return at_ms;
    };
    // Re-enter through `day_start_ms` so the DST-gap handling above is single-sourced.
    let Some(noon) = date.and_hms_opt(12, 0, 0) else {
        return at_ms;
    };
    let anchor = tz
        .from_local_datetime(&noon)
        .earliest()
        .map(|d| d.timestamp_millis())
        .unwrap_or(at_ms);
    day_start_ms(tz, anchor)
}

// ── Subject ────────────────────────────────────────────────────────────────────

/// One body of material being learned: a course, a paper, a chapter.
///
/// Subjects are this app's top-level scope. There is deliberately no workspace
/// above them — Tuition is single-learner by design (no roster, no billing, no
/// parent portal), so a second scope level would be a table every query joined
/// through and nothing would ever put a second row in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Subject {
    pub id: String,
    pub name: String,
    /// Free text the list view shows under the name.
    #[serde(default)]
    pub detail: Option<String>,
    /// `YYYY-MM-DD`, or `None` for no exam.
    ///
    /// Authoritative here rather than in [`TuitionSettings`], because the
    /// `goal.at-risk` event is per subject and its contract is "does NOT fire when
    /// the subject has no exam date set" — a single node-level preference cannot
    /// express that. The `tuition-exam-date` preference in the settings tab is the
    /// *default offered* for a new subject, not the value the projection reads.
    #[serde(default)]
    pub exam_date: Option<String>,
    /// IANA zone the day boundaries of this subject's schedule are measured in.
    pub timezone: String,
    pub created_at: i64,
    pub updated_at: i64,
}

impl Subject {
    pub fn tz(&self) -> chrono_tz::Tz {
        resolve_tz(&self.timezone)
    }
}

// ── Source ─────────────────────────────────────────────────────────────────────

/// Where a source document came from. Determines which ingest path runs, and is
/// shown as provenance next to every item generated from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// A file on this node, parsed through the `document.parse` capability.
    File,
    /// A URL fetched then parsed.
    Url,
    /// Text pasted straight into the companion; no parser involved.
    Text,
}

impl SourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Url => "url",
            Self::Text => "text",
        }
    }

    /// Unknown values degrade to `Text` — the kind that needs no parser, so a
    /// corrupt row still renders its stored text instead of failing the list.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "file" => Self::File,
            "url" => Self::Url,
            _ => Self::Text,
        }
    }
}

/// A document the subject's skills and items were derived from.
///
/// `parsed_text` is stored, not re-derived: the parse runs through whichever
/// `document.parse` provider happens to be installed, so re-parsing the same file
/// six months later can legitimately produce different text — and every item
/// citing `source_ref` would then point at an offset that no longer exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub subject_id: String,
    pub kind: SourceKind,
    pub title: String,
    /// Path or URL, depending on `kind`. `None` for pasted text.
    #[serde(default)]
    pub uri: Option<String>,
    pub parsed_text: String,
    /// Which `document.parse` provider produced `parsed_text` (`docling`,
    /// `markitdown`, …), or `None` when the text needed no parser.
    #[serde(default)]
    pub parser: Option<String>,
    pub created_at: i64,
}

// ── Skill ──────────────────────────────────────────────────────────────────────

/// The four Bayesian Knowledge Tracing parameters, all in `[0,1]`.
///
/// Per skill rather than global, because a skill whose items are 2-choice has a
/// structurally different guess rate from one whose items are free response, and
/// grading the first with the second's numbers is how a mastery model lies to you.
/// The defaults are the standard BKT starting point and are what a proposed skill
/// is seeded with.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BktParams {
    /// P(already knew it before any practice).
    pub p_init: f64,
    /// P(learn it on an attempt, given not known).
    pub p_transit: f64,
    /// P(get it wrong although known).
    pub p_slip: f64,
    /// P(get it right although not known).
    pub p_guess: f64,
}

pub const DEFAULT_P_INIT: f64 = 0.20;
pub const DEFAULT_P_TRANSIT: f64 = 0.15;
pub const DEFAULT_P_SLIP: f64 = 0.10;
pub const DEFAULT_P_GUESS: f64 = 0.20;

impl Default for BktParams {
    fn default() -> Self {
        Self {
            p_init: DEFAULT_P_INIT,
            p_transit: DEFAULT_P_TRANSIT,
            p_slip: DEFAULT_P_SLIP,
            p_guess: DEFAULT_P_GUESS,
        }
    }
}

impl BktParams {
    /// Clamp every parameter into `[0,1]`. Applied on the write path so a
    /// hand-edited row (or a model-proposed one) cannot produce a posterior update
    /// that leaves the unit interval and poisons every later attempt.
    pub fn clamped(self) -> Self {
        fn unit(v: f64) -> f64 {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
        Self {
            p_init: unit(self.p_init),
            p_transit: unit(self.p_transit),
            p_slip: unit(self.p_slip),
            p_guess: unit(self.p_guess),
        }
    }
}

/// Whether a skill is part of the deck yet.
///
/// The model-proposed half of ingest lands as `Proposed` and is invisible to
/// scheduling until a person promotes it. That review step is the reason this app
/// can claim its mastery numbers mean something.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillStatus {
    Proposed,
    Active,
    Archived,
}

impl SkillStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Active => "active",
            Self::Archived => "archived",
        }
    }

    /// Unknown values degrade to `Proposed`: the state that is *not* scheduled and
    /// not graded. Guessing `Active` for an unreadable row would put something the
    /// learner never accepted into the deck.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "active" => Self::Active,
            "archived" => Self::Archived,
            _ => Self::Proposed,
        }
    }
}

pub const DEFAULT_EASE: f64 = 2.5;
/// SM-2's floor. An ease below this makes intervals collapse to nothing and the
/// skill reappears every single day forever.
pub const MIN_EASE: f64 = 1.3;

/// One learnable thing, carrying both halves of its state: what the learner knows
/// (the BKT posterior) and when it is next owed (the SM-2 schedule).
///
/// These are kept on one row rather than split into a `mastery` and a `schedule`
/// table because every read wants both — "what should I study next" is
/// `min(mastery)` over `due_at <= now` with mastered prerequisites — and a join
/// per skill for a page that always shows both buys nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub subject_id: String,
    pub name: String,
    #[serde(default)]
    pub detail: Option<String>,
    pub status: SkillStatus,
    /// The source this skill was proposed from, when it came from ingest.
    #[serde(default)]
    pub source_id: Option<String>,
    pub params: BktParams,
    /// The running posterior `P(known)`, seeded to `params.p_init`.
    pub mastery: f64,
    pub ease: f64,
    pub interval_days: u32,
    pub reps: u32,
    pub lapses: u32,
    /// Epoch millis at a day boundary in the subject's timezone. `None` until the
    /// skill has been reviewed once — a never-studied skill is not "overdue".
    #[serde(default)]
    pub due_at: Option<i64>,
    #[serde(default)]
    pub last_reviewed_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A prerequisite edge: `prereq_id` must be mastered before `skill_id` is ready.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrereqEdge {
    pub skill_id: String,
    pub prereq_id: String,
}

// ── Items ──────────────────────────────────────────────────────────────────────

/// The five item kinds. The first four are graded by comparison with **no model in
/// the loop at all**; only [`ItemKind::Free`] reaches one, and it shows the rubric
/// it was graded against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Mcq,
    Cloze,
    Numeric,
    Exact,
    Free,
}

impl ItemKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mcq => "mcq",
            Self::Cloze => "cloze",
            Self::Numeric => "numeric",
            Self::Exact => "exact",
            Self::Free => "free",
        }
    }

    /// Unknown values degrade to `Free`, the kind that is never auto-graded. An
    /// unreadable kind must not become one arithmetic decides on.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "mcq" => Self::Mcq,
            "cloze" => Self::Cloze,
            "numeric" => Self::Numeric,
            "exact" => Self::Exact,
            _ => Self::Free,
        }
    }

    /// Whether grading this kind is pure comparison.
    pub const fn is_objective(self) -> bool {
        !matches!(self, Self::Free)
    }
}

/// One selectable answer of an [`ItemKind::Mcq`]. `id` is what an attempt records,
/// so re-ordering or re-wording choices never invalidates stored attempts.
// `ToSchema` because this rides inside the `/items` request bodies, which Core lowers
// into an LLM tool: without it the `choices` argument has no shape and a model writing
// a multiple-choice item has to guess the field names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Choice {
    pub id: String,
    pub text: String,
}

/// How far off a numeric answer may be and still count.
///
/// Tagged, never a bare number: `± 0.5` and `± 0.5 %` are different questions, and
/// a stored tolerance that does not say which one it is has to be guessed at grade
/// time. Both bounds are decimal *strings* for the same reason the expected value
/// is — see [`AnswerKey::Numeric`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Tolerance {
    /// `|given - expected| <= value`.
    Absolute { value: String },
    /// `|given - expected| <= |expected| * value / 100`.
    RelativePercent { value: String },
}

/// The gradeable half of an item, stored as a JSON blob in `items.answer`.
///
/// A tagged enum rather than a column per kind, because the alternative is six
/// mostly-NULL columns whose valid combinations exist only in a comment. This way
/// the grader gets a total `match` and an item that is missing its answer cannot be
/// constructed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AnswerKey {
    Mcq {
        choice_id: String,
    },
    /// One entry per blank, each holding the accepted alternatives for that blank.
    Cloze {
        blanks: Vec<Vec<String>>,
    },
    /// `expected` is a decimal **string**, parsed by this crate's fixed-point
    /// decimal at grade time. It is deliberately not an `f64`: `f64::from_str` must
    /// never be what decides whether an answer was correct.
    Numeric {
        expected: String,
        // Inlined for the same reason the `answer` field itself is: Core resolves a
        // `$ref` only at the top of a schema node, so a pointer nested two levels
        // deep inside this variant reaches the model as an opaque token and the
        // tolerance shape becomes unguessable.
        #[schema(inline)]
        tolerance: Tolerance,
    },
    Exact {
        text: String,
        #[serde(default)]
        alternatives: Vec<String>,
    },
    /// The only kind a model grades. The rubric lives here rather than in its own
    /// column so it is impossible to have a free-response item with no rubric.
    Free {
        rubric: String,
    },
    /// A stored answer whose JSON did not decode. Never write this: it is a read-path
    /// placeholder, and the grader refuses to grade it.
    // Present so one corrupt row cannot fail a whole list query — but it is a dead
    // end on purpose: `AnswerKey::encode` refuses to write it back, so a
    // read-modify-write through this variant cannot overwrite a recoverable answer
    // with the shell of it. The `///` above is deliberately terse and imperative
    // because it now reaches a model deciding what to send; this paragraph does not.
    Malformed {
        raw: String,
    },
}

impl AnswerKey {
    /// Tolerant decode for the read path.
    pub fn decode(raw: &str) -> Self {
        serde_json::from_str(raw).unwrap_or_else(|_| Self::Malformed {
            raw: raw.to_string(),
        })
    }

    /// Encode for the write path. Fails on [`AnswerKey::Malformed`] — including one
    /// a client synthesized, since the wire form is symmetric — which is what keeps
    /// the variant from ever reaching the database.
    pub fn encode(&self) -> anyhow::Result<String> {
        if matches!(self, Self::Malformed { .. }) {
            anyhow::bail!("an item's answer key is unreadable and cannot be rewritten");
        }
        Ok(serde_json::to_string(self)?)
    }

    /// The item kind this key grades. `items.kind` is stored as its own column for
    /// filtering, and this is what the write path checks it against — a `numeric`
    /// item carrying an `mcq` key would be ungradeable in a way nothing detects
    /// until the learner is staring at it.
    pub const fn kind(&self) -> Option<ItemKind> {
        match self {
            Self::Mcq { .. } => Some(ItemKind::Mcq),
            Self::Cloze { .. } => Some(ItemKind::Cloze),
            Self::Numeric { .. } => Some(ItemKind::Numeric),
            Self::Exact { .. } => Some(ItemKind::Exact),
            Self::Free { .. } => Some(ItemKind::Free),
            Self::Malformed { .. } => None,
        }
    }
}

/// Who wrote an item. Rendered next to it, because "a model wrote this question"
/// is information the learner is entitled to when the question looks wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemOrigin {
    /// Generated from source material by the generation model.
    Model,
    /// Written by the learner in the companion.
    Human,
    /// Promoted from a Study-mode review candidate.
    Candidate,
}

impl ItemOrigin {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Human => "human",
            Self::Candidate => "candidate",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw {
            "human" => Self::Human,
            "candidate" => Self::Candidate,
            _ => Self::Model,
        }
    }
}

/// A practice item. Stored, versioned and reusable — generation is the expensive
/// half of this app and re-asking a known-good question costs nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    /// Denormalized from the skill so subject-scoped reads and the delete cascade
    /// are single-table. There is no FK to keep it honest, so it is written once
    /// from the skill at insert and never edited.
    pub subject_id: String,
    pub skill_id: String,
    pub kind: ItemKind,
    pub prompt: String,
    /// Only populated for [`ItemKind::Mcq`].
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub answer: AnswerKey,
    pub origin: ItemOrigin,
    /// The model that generated it, when `origin` is `Model`.
    #[serde(default)]
    pub origin_model: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    /// A quotation or locator into the source, shown so a disputed answer can be
    /// checked against the material rather than argued with.
    #[serde(default)]
    pub source_ref: Option<String>,
    /// Bumped on every edit. Attempts record the version they were graded under, so
    /// a rewritten question does not retroactively change what a past answer meant.
    pub version: i64,
    pub archived: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

// ── Attempts ───────────────────────────────────────────────────────────────────

/// What decided a grade. The whole trust story of this app is that the first value
/// covers four of the five item kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradedBy {
    /// Pure comparison. No model was consulted.
    Deterministic,
    /// Marked by the grading model against the item's written rubric.
    Rubric,
}

impl GradedBy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Rubric => "rubric",
        }
    }

    /// Unknown values degrade to `Rubric` — the *weaker* claim. Reporting a grade
    /// as deterministic when the row is unreadable would be the one lie this app
    /// cannot afford.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "deterministic" => Self::Deterministic,
            _ => Self::Rubric,
        }
    }
}

/// One graded (or awaiting-grade) answer.
///
/// `correct` is `None` while a free-response answer is still awaiting its rubric
/// mark. That state is load-bearing: `mastery.dropped` must not fire on an answer
/// that has not been graded yet, and the posterior must not move either — which is
/// why `mastery_before` and `mastery_after` are both recorded on the row rather
/// than recomputed from the skill later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub id: String,
    pub subject_id: String,
    pub skill_id: String,
    pub item_id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    /// The item version this was graded against.
    pub item_version: i64,
    pub response: String,
    #[serde(default)]
    pub correct: Option<bool>,
    /// Partial credit in `[0,1]` where the kind supports it (cloze blanks, a
    /// rubric mark). `None` for a pass/fail grade.
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub graded_by: Option<GradedBy>,
    /// The rubric mark's reasoning, shown with the grade.
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub latency_ms: Option<i64>,
    pub mastery_before: f64,
    pub mastery_after: f64,
    /// `false` when the BKT denominator was degenerate and the posterior was left
    /// unchanged. Recorded rather than dropped so "why did nothing move" has an
    /// answer, and so this attempt is excluded from any rate computed over history.
    pub informative: bool,
    pub at: i64,
}

// ── Sessions ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Planned but not started. The plan is stored before the first question is
    /// shown so a reload resumes the same session rather than re-planning against
    /// a schedule the first answer already moved.
    Planned,
    Active,
    Finished,
    Abandoned,
}

impl SessionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Active => "active",
            Self::Finished => "finished",
            Self::Abandoned => "abandoned",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw {
            "active" => Self::Active,
            "finished" => Self::Finished,
            "abandoned" => Self::Abandoned,
            _ => Self::Planned,
        }
    }
}

/// One planned study sitting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub subject_id: String,
    /// The budget the planner filled.
    pub planned_minutes: u32,
    pub status: SessionStatus,
    /// Sum of `mastery_after - mastery_before` over the session's informative
    /// attempts, written at finish.
    ///
    /// Stored rather than recomputed because the trajectory projection reads the
    /// trailing ten sessions on every tick, and recomputing means joining every
    /// attempt of every one of them each time — for a number that can never change
    /// once the session is closed.
    #[serde(default)]
    pub mastery_gain: Option<f64>,
    #[serde(default)]
    pub summary: Option<String>,
    pub created_at: i64,
    #[serde(default)]
    pub started_at: Option<i64>,
    #[serde(default)]
    pub finished_at: Option<i64>,
}

/// One item on a session's plan, in the order the planner chose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionItem {
    pub session_id: String,
    pub position: u32,
    pub item_id: String,
    pub skill_id: String,
    /// The planner's own estimates, kept so a finished session can be compared
    /// against what it predicted — the planner is only as good as its cost model
    /// and this is the only record of what that model said.
    pub est_cost_ms: i64,
    pub est_gain: f64,
    /// Set when the item is answered. `None` means still pending.
    #[serde(default)]
    pub attempt_id: Option<String>,
}

// ── Review candidates ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Pending,
    Accepted,
    Rejected,
}

impl CandidateStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }

    pub fn parse(raw: &str) -> Self {
        match raw {
            "accepted" => Self::Accepted,
            "rejected" => Self::Rejected,
            _ => Self::Pending,
        }
    }
}

/// A fact a chat turn taught, queued by the Study-mode hook and waiting on a
/// person.
///
/// Never auto-accepted, and the reason is not politeness: a deck you did not choose
/// is a deck you stop trusting, and the posterior computed over it is then
/// worthless. Accepting one is what turns it into a [`Skill`] + [`Item`] pair.
///
/// `source_key` / `source_index` are the hook's KV key and the candidate's offset
/// within that key's payload. They exist for exactly one reason: the sidecar's
/// drain is `keys` → `get` → `delete`, and a crash between the `get` and the
/// `delete` re-delivers the whole payload on the next tick. A UNIQUE index over the
/// pair makes the re-delivery a no-op instead of a duplicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewCandidate {
    pub id: String,
    pub subject_id: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    pub prompt: String,
    pub answer: String,
    /// The short topic label the hook asked the model for, reused verbatim across
    /// candidates on the same topic. Acceptance matches it against existing skill
    /// names, which is why `skills(subject_id, name)` is UNIQUE.
    #[serde(default)]
    pub skill_label: Option<String>,
    pub status: CandidateStatus,
    pub source_key: String,
    pub source_index: i64,
    /// The item acceptance created, once decided.
    #[serde(default)]
    pub item_id: Option<String>,
    pub created_at: i64,
    #[serde(default)]
    pub decided_at: Option<i64>,
}

/// One candidate as it arrives from the hook's KV payload, before it has an id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewCandidate {
    pub prompt: String,
    pub answer: String,
    #[serde(default)]
    pub skill: Option<String>,
}

/// The whole value stored under one `candidate:<subject>:<conversation>:<unique>`
/// key by `hooks/study.js`.
///
/// The field names here are the hook's, byte for byte — it writes this shape and
/// this decodes it, with nothing in between able to reconcile a mismatch. `v` is
/// the hook's own payload version; an unrecognized one is skipped rather than
/// guessed at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEnvelope {
    #[serde(default = "one")]
    pub v: i64,
    pub subject_id: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub candidates: Vec<NewCandidate>,
}

const fn one() -> i64 {
    1
}

/// The payload version `hooks/study.js` writes today.
pub const CANDIDATE_ENVELOPE_VERSION: i64 = 1;

// ── Settings ───────────────────────────────────────────────────────────────────

/// Node-level knobs of the deterministic spine.
///
/// Deliberately holds nothing that has a `pref_key` in the manifest's settings tab
/// (`tuition-active-subject`, the two model pickers, `tuition-daily-minutes`,
/// `tuition-exam-date`). Those live in Core's preference store, which is where the
/// user edits them; duplicating one here would give a single number two sources of
/// truth and no rule for which wins. The session budget accordingly arrives in the
/// `/sessions/plan` body rather than being read from here.
// `ToSchema` because this IS the `PUT /api/tuition/settings` body — every field is a
// scalar, so it lowers into a flat, fully visible argument list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TuitionSettings {
    /// The posterior at which a prerequisite counts as mastered, gating
    /// `ready_skills`. Also the threshold `mastery.dropped` fires on crossing.
    pub ready_threshold: f64,
    /// The posterior the "how many more correct" projection aims at.
    pub target_mastery: f64,
    /// Cap on items from one skill in a single session, so one weak skill cannot
    /// consume the whole sitting.
    pub per_skill_item_cap: u32,
    /// Seed for the planner's cost model, used until a skill has observed
    /// latencies to take a median over.
    pub default_item_seconds: u32,
    /// Sessions the learner expects to do per day, converting days-to-exam into
    /// sessions-available for the trajectory projection.
    pub sessions_per_day: f64,
    /// How far over the available sessions the projection may run before
    /// `goal.at-risk` fires. 1.15 = 15% overrun.
    pub at_risk_overrun: f64,
    /// Trailing sessions the observed learning rate is averaged over.
    pub trajectory_window: u32,
    /// Below this many finished sessions the trajectory reports `unknown` rather
    /// than extrapolating.
    pub trajectory_min_sessions: u32,
}

impl Default for TuitionSettings {
    fn default() -> Self {
        Self {
            ready_threshold: 0.80,
            target_mastery: 0.90,
            per_skill_item_cap: 5,
            default_item_seconds: 60,
            sessions_per_day: 1.0,
            at_risk_overrun: 1.15,
            trajectory_window: 10,
            trajectory_min_sessions: 3,
        }
    }
}

// ── Write-path inputs ──────────────────────────────────────────────────────────
//
// The fields a caller supplies, as opposed to the ones the store derives. The
// split is not ceremony: `subject_id` on an item and `item_version` on an attempt
// are denormalized copies with no foreign key to keep them honest, so they are
// read off the parent row by the store and are deliberately NOT accepted from a
// caller who could get them wrong.

/// A practice item as it arrives from generation or from the companion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewItem {
    pub skill_id: String,
    pub kind: ItemKind,
    pub prompt: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub answer: AnswerKey,
    #[serde(default = "default_origin")]
    pub origin: ItemOrigin,
    #[serde(default)]
    pub origin_model: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub source_ref: Option<String>,
}

/// An item posted with no stated origin came from a person typing it.
fn default_origin() -> ItemOrigin {
    ItemOrigin::Human
}

/// A graded (or awaiting-grade) answer as it arrives from the session runner, the
/// `tuition__log` MCP tool, or the grader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewAttempt {
    pub skill_id: String,
    pub item_id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    pub response: String,
    #[serde(default)]
    pub correct: Option<bool>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub graded_by: Option<GradedBy>,
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub latency_ms: Option<i64>,
    pub mastery_before: f64,
    pub mastery_after: f64,
    pub informative: bool,
}

/// One line of a session plan, in the order the planner chose. Positions are
/// assigned by the store from the slice order, so a plan cannot arrive with two
/// items at the same position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedItem {
    pub item_id: String,
    pub skill_id: String,
    pub est_cost_ms: i64,
    pub est_gain: f64,
}

// ── Health ─────────────────────────────────────────────────────────────────────

/// What `/health` reports. Counts, not liveness: the probe has to prove the
/// database is readable, because a sidecar that answers 200 with an unopenable DB
/// is the exact failure Core's supervisor would otherwise never notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreCounts {
    pub subjects: i64,
    pub skills: i64,
    pub items: i64,
    pub attempts: i64,
    pub pending_candidates: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_carry_their_prefix_and_do_not_repeat() {
        let a = new_id(ID_SKILL);
        let b = new_id(ID_SKILL);
        assert!(a.starts_with("skl_"));
        assert_ne!(a, b);
    }

    #[test]
    fn day_start_is_local_midnight_not_utc_midnight() {
        let tz = resolve_tz("Asia/Tokyo");
        // 2026-08-10T23:58:00+09:00 == 2026-08-10T14:58:00Z. A session at 23:58
        // local must still belong to the 10th, whose start is 2026-08-09T15:00:00Z.
        let at = 1_786_373_880_000; // 2026-08-10T14:58:00Z
        let start = day_start_ms(tz, at);
        assert!(start <= at);
        assert!(at - start < 24 * 3_600_000);
        // Same instant in UTC lands on a different day boundary — which is the
        // whole reason the subject carries a timezone.
        assert_ne!(start, day_start_ms(resolve_tz("UTC"), at));
    }

    #[test]
    fn day_start_is_idempotent() {
        let tz = resolve_tz("Europe/Berlin");
        let at = now_ms();
        let once = day_start_ms(tz, at);
        assert_eq!(once, day_start_ms(tz, once));
    }

    #[test]
    fn adding_days_lands_on_a_day_boundary_across_a_dst_change() {
        // Europe/Berlin springs forward on 2026-03-29. A one-day step across it
        // must still land on local midnight, not 23:00 or 01:00 of the wrong day.
        let tz = resolve_tz("Europe/Berlin");
        let before = day_start_ms(tz, 1_774_494_000_000); // 2026-03-26T03:00:00Z
        let after = day_start_plus_days(tz, before, 4);
        assert_eq!(after, day_start_ms(tz, after));
        let span_hours = (after - before) / 3_600_000;
        // Four days minus the hour DST ate.
        assert_eq!(span_hours, 95);
    }

    #[test]
    fn unknown_timezone_falls_back_to_utc_instead_of_failing() {
        assert_eq!(resolve_tz("Middle/Earth"), chrono_tz::UTC);
        assert_eq!(resolve_tz(""), chrono_tz::UTC);
    }

    #[test]
    fn answer_keys_round_trip_and_carry_their_kind() {
        let key = AnswerKey::Numeric {
            expected: "9.81".into(),
            tolerance: Tolerance::RelativePercent { value: "1" .into() },
        };
        let encoded = key.encode().unwrap();
        assert_eq!(AnswerKey::decode(&encoded), key);
        assert_eq!(key.kind(), Some(ItemKind::Numeric));
    }

    #[test]
    fn a_malformed_answer_decodes_but_can_never_be_written_back() {
        let key = AnswerKey::decode("{ not json");
        assert!(matches!(key, AnswerKey::Malformed { .. }));
        assert_eq!(key.kind(), None);
        // The guard that matters: a read-modify-write through this variant must not
        // overwrite the recoverable original with the shell of it.
        assert!(key.encode().is_err());
    }

    #[test]
    fn enum_columns_degrade_to_the_safe_value_never_the_flattering_one() {
        // An unreadable skill is NOT scheduled…
        assert_eq!(SkillStatus::parse("beleived"), SkillStatus::Proposed);
        // …an unreadable item kind is NOT auto-graded…
        assert_eq!(ItemKind::parse("mcqq"), ItemKind::Free);
        // …and an unreadable grade does NOT claim to be arithmetic.
        assert_eq!(GradedBy::parse(""), GradedBy::Rubric);
        assert_eq!(CandidateStatus::parse("?"), CandidateStatus::Pending);
        assert_eq!(SessionStatus::parse("?"), SessionStatus::Planned);
        assert_eq!(SourceKind::parse("?"), SourceKind::Text);
        assert_eq!(ItemOrigin::parse("?"), ItemOrigin::Model);
    }

    #[test]
    fn enum_wire_and_sql_strings_are_the_same_string() {
        // `as_str` is the SQL value and serde's `rename_all` output is the wire
        // value; they must not be allowed to drift apart.
        for kind in [
            ItemKind::Mcq,
            ItemKind::Cloze,
            ItemKind::Numeric,
            ItemKind::Exact,
            ItemKind::Free,
        ] {
            let wire = serde_json::to_string(&kind).unwrap();
            assert_eq!(wire, format!("\"{}\"", kind.as_str()));
            assert_eq!(ItemKind::parse(kind.as_str()), kind);
        }
        for status in [
            SkillStatus::Proposed,
            SkillStatus::Active,
            SkillStatus::Archived,
        ] {
            assert_eq!(
                serde_json::to_string(&status).unwrap(),
                format!("\"{}\"", status.as_str())
            );
            assert_eq!(SkillStatus::parse(status.as_str()), status);
        }
    }

    #[test]
    fn bkt_params_are_clamped_into_the_unit_interval() {
        let wild = BktParams {
            p_init: 1.7,
            p_transit: -0.2,
            p_slip: f64::NAN,
            p_guess: 0.25,
        }
        .clamped();
        assert_eq!(wild.p_init, 1.0);
        assert_eq!(wild.p_transit, 0.0);
        assert_eq!(wild.p_slip, 0.0);
        assert_eq!(wild.p_guess, 0.25);
    }

    #[test]
    fn the_hook_payload_decodes_as_the_hook_writes_it() {
        // Copied from the `host.storage.set` call in `apps-store/tuition/hooks/study.js`.
        // If this stops matching, the drain silently files nothing.
        let raw = r#"{
          "v": 1,
          "subject_id": "sub_abc",
          "conversation_id": "conv_1",
          "agent_id": null,
          "created_at": "2026-08-10T06:00:00.000Z",
          "candidates": [
            { "prompt": "What does a beta-blocker do?", "answer": "Blocks beta-adrenergic receptors.", "skill": "Beta blockers" }
          ]
        }"#;
        let env: CandidateEnvelope = serde_json::from_str(raw).unwrap();
        assert_eq!(env.v, CANDIDATE_ENVELOPE_VERSION);
        assert_eq!(env.subject_id, "sub_abc");
        assert_eq!(env.candidates.len(), 1);
        assert_eq!(env.candidates[0].skill.as_deref(), Some("Beta blockers"));
    }

    #[test]
    fn settings_defaults_match_the_documented_spine_constants() {
        let s = TuitionSettings::default();
        assert_eq!(s.ready_threshold, 0.80);
        assert_eq!(s.target_mastery, 0.90);
        assert_eq!(s.per_skill_item_cap, 5);
        assert_eq!(s.trajectory_min_sessions, 3);
        // Round-trips through the JSON column it is stored in.
        let encoded = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<TuitionSettings>(&encoded).unwrap(), s);
    }
}
