//! SQLite persistence for the whole Tuition spine (`~/.ryu/tuition.db`).
//!
//! ## What lives here and what does not
//!
//! Every SQL statement in this crate is in this file, one method per operation.
//! The arithmetic — the BKT update, the SM-2 step, the planner's knapsack, the
//! trajectory projection — deliberately is not: those are pure functions of stored
//! state, and keeping them out of the persistence layer is what lets them be tested
//! against a table of inputs with no database at all. The store's job is to hand
//! them their inputs and durably record what they returned, including the
//! `mastery_before` / `mastery_after` pair on every attempt so a posterior can be
//! audited backwards.
//!
//! ## No foreign keys — a deliberate choice, not an omission
//!
//! `PRAGMA foreign_keys` is **per-connection and not persisted in the file**, so a
//! schema with real `ON DELETE CASCADE` behaves differently depending on which code
//! path opened the connection — silent orphans on one, cascades on the other. That
//! failure mode is invisible until data is already lost. Instead, deletes run an
//! explicit ordered cascade inside a transaction (see
//! [`TuitionStore::delete_subject`]), which is auditable and connection-independent.
//!
//! The corollary is that `subject_id` is denormalized onto `items`, `attempts` and
//! `skill_prereqs` rather than being reached through `skills`. It makes every
//! cascade a single-table delete and every subject-scoped list an index seek, and
//! the columns are written once by this module from the parent row — never accepted
//! from a caller.
//!
//! ## Uniqueness is enforced by indexes, never by a pre-insert SELECT
//!
//! Three of them carry real weight:
//!
//! - `skills(subject_id, name)` — the Study-mode hook reuses its short skill labels
//!   verbatim across candidates on one topic, so acceptance is an upsert on this
//!   index. A read-then-insert would let two candidates accepted in the same second
//!   create two skills with the same name and split one posterior in half.
//! - `review_candidates(source_key, source_index)` — the drain is `storage.keys` →
//!   `get` → `delete`, and a crash between the `get` and the `delete` re-delivers
//!   the whole payload next tick. This index turns that into a no-op.
//! - `event_marks(kind, subject_id, ref_id)` — the once-per-cooldown guard behind
//!   every declared hook event. `ref_id` is `NOT NULL DEFAULT ''`, because SQLite
//!   does not consider two NULLs equal and a nullable column here would silently
//!   permit unlimited duplicate marks — i.e. no guard at all.
//!
//! ## Composing the multi-write operations
//!
//! Two flows span several methods, and nothing binds them into one transaction —
//! the store deliberately does not own the arithmetic that sits between the writes.
//! So the ORDER is part of the contract, and both orders are chosen so that a crash
//! (or a lost CAS) leaves state that is merely incomplete rather than wrong:
//!
//! **Answering an item.** [`TuitionStore::record_attempt`] →
//! [`TuitionStore::bind_session_attempt`] → and only if that returned `true`,
//! [`TuitionStore::update_skill_mastery`] / [`TuitionStore::update_skill_schedule`]
//! / [`TuitionStore::clear_event_mark`]. Recording first is forced (the bind needs
//! an attempt id), and the bind is the claim: a `false` means this slot was already
//! answered, so the posterior updates MUST be skipped. An attempt row that lost the
//! race is harmless — it carries its own `mastery_before`/`mastery_after` and moved
//! nothing. Applying the posterior first and binding second would double-count a
//! resubmitted answer, which is the one failure this ordering exists to prevent.
//! Outside a session (the `tuition__log` MCP tool) the bind step is simply absent.
//!
//! **Accepting a review candidate.** [`TuitionStore::create_item`] →
//! [`TuitionStore::decide_candidate`], and on a `false` from decide (someone else
//! decided it first) delete the item just created. The reverse order cannot be
//! used, because `decide_candidate` records the id of the item it produced.
//!
//! ## Locking
//!
//! One `Arc<tokio::sync::Mutex<Connection>>` (the async mutex, matching
//! `ryu-social` / `ryu-teams`) — a single writer with WAL underneath.
//! `busy_timeout` still matters because WAL admits readers from OTHER processes (a
//! `sqlite3` shell, a backup), about which this process's mutex knows nothing.
//!
//! Everything that walks the prerequisite graph does so under that same mutex, so
//! every such walk is bounded — an unbounded frontier over a cyclic edge set would
//! not merely loop, it would hold the connection and hang every request in the
//! process, `/health` included.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Row};
use tokio::sync::Mutex;

use crate::models::*;

/// The schema version this build expects. Bump it and add a `current < N` arm in
/// [`TuitionStore::migrate`] when the shape changes.
///
/// A `PRAGMA user_version` ladder rather than bare `CREATE TABLE IF NOT EXISTS`:
/// `IF NOT EXISTS` cannot add a COLUMN to a table that already exists, so the
/// moment a later change needs one it would have to retrofit the whole versioning
/// scheme onto live user databases. Paying for it now costs one integer.
const SCHEMA_VERSION: i32 = 1;

/// The single row [`TuitionStore::get_settings`] reads. Node-level, not per
/// subject: these are knobs of the spine's arithmetic, and a learner running two
/// subjects does not want two definitions of "mastered".
const SETTINGS_SCOPE: &str = "node";

/// Ceiling on how many list rows any one call may return, and what it returns when
/// the caller states no preference. Applied in the store rather than only in the
/// handlers so an MCP tool call cannot ask for a million rows either.
pub const MAX_LIMIT: i64 = 500;
pub const DEFAULT_LIMIT: i64 = 200;

/// Hard cap on nodes visited by the prerequisite-cycle walk.
///
/// The walk already carries a visited set, so this cannot trip on a well-formed
/// graph — it exists for the graph that is already cyclic when we meet it (an
/// older build, a `sqlite3` shell edit), where "already broken" must degrade to a
/// refused write and not a hung process.
const MAX_PREREQ_WALK: usize = 10_000;

fn clamp_limit(limit: i64) -> i64 {
    if limit <= 0 {
        DEFAULT_LIMIT
    } else {
        limit.min(MAX_LIMIT)
    }
}

/// SQLite-backed store for the Tuition spine. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub struct TuitionStore {
    conn: Arc<Mutex<Connection>>,
}

impl TuitionStore {
    /// Open (creating if needed) the DB at `path` and migrate it. The path is
    /// injected by the caller (`paths::ryu_dir().join("tuition.db")`) so this module
    /// has no opinion about where the node's data lives.
    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).context("creating parent dir for tuition.db")?;
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("opening tuition db at {}", path.display()))?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// In-memory store. A plain `pub fn`, not `#[cfg(test)]`, so the later modules'
    /// tests (the graders, the planner, the MCP server) can build a real store
    /// without a temp file — the same convention `ryu-social` and `ryu-teams` use.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::prepare(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Pragmas then migrations. Both paths call this so an in-memory store is
    /// byte-for-byte the same schema as a real one — a divergence here would make
    /// every module test a lie.
    fn prepare(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            // WAL: readers never block the single writer, which matters because the
            // tick task writes while the companion polls the due list.
            // synchronous=NORMAL: safe under WAL (a crash can lose the last commit,
            // not corrupt the file) and avoids an fsync per graded answer.
            // busy_timeout: this process serializes its own writes behind the mutex,
            // but another process holding the file (a shell, a backup) would
            // otherwise fail instantly instead of waiting.
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA busy_timeout = 5000;",
        )
        .context("applying tuition db pragmas")?;
        Self::migrate(conn)
    }

    /// The `PRAGMA user_version` ladder.
    ///
    /// Every arm must be safe to re-run, because this runs on EVERY open. That is
    /// not a style preference: an arm that can fail (a `CREATE UNIQUE INDEX` over
    /// rows that already violate it, say) is not a one-time error — the sidecar
    /// would refuse to boot, forever, on exactly the databases the fix was meant to
    /// repair. So every statement in every `V*_DDL` is `IF NOT EXISTS`-guarded, and
    /// any future arm that adds a constraint must delete the rows violating it
    /// first.
    fn migrate(conn: &Connection) -> Result<()> {
        let current: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if current >= SCHEMA_VERSION {
            return Ok(());
        }
        if current < 1 {
            conn.execute_batch(V1_DDL)
                .context("applying tuition schema v1")?;
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)
            .context("stamping tuition schema version")?;
        Ok(())
    }
}

/// The complete v1 schema.
///
/// Collapsed into ONE statement batch rather than replayed as a migration history,
/// because there are no existing databases to migrate — this app has never shipped.
/// Every table is declared in its final shape.
const V1_DDL: &str = "
CREATE TABLE IF NOT EXISTS subjects (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  detail     TEXT,
  -- A `YYYY-MM-DD` string, not an instant: the learner typed a calendar date and
  -- it must not shift by a day when read in another zone.
  exam_date  TEXT,
  timezone   TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
-- Deliberately NOT unique. Nothing in this crate depends on subject names being
-- distinct, and a UNIQUE here would turn 'I made two subjects called Anatomy' into
-- a constraint violation surfacing as a 500.
CREATE INDEX IF NOT EXISTS idx_subjects_name ON subjects(name);

CREATE TABLE IF NOT EXISTS sources (
  id          TEXT PRIMARY KEY,
  subject_id  TEXT NOT NULL,
  kind        TEXT NOT NULL,
  title       TEXT NOT NULL,
  uri         TEXT,
  -- The parse output is STORED, not re-derived. `document.parse` resolves to
  -- whichever provider happens to be installed, so re-parsing the same file later
  -- can legitimately produce different text — and every item citing a `source_ref`
  -- into it would then point at something that no longer exists.
  parsed_text TEXT NOT NULL,
  parser      TEXT,
  created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_subject ON sources(subject_id, created_at DESC);

CREATE TABLE IF NOT EXISTS skills (
  id               TEXT PRIMARY KEY,
  subject_id       TEXT NOT NULL,
  name             TEXT NOT NULL,
  detail           TEXT,
  status           TEXT NOT NULL,
  source_id        TEXT,
  p_init           REAL NOT NULL,
  p_transit        REAL NOT NULL,
  p_slip           REAL NOT NULL,
  p_guess          REAL NOT NULL,
  mastery          REAL NOT NULL,
  ease             REAL NOT NULL,
  interval_days    INTEGER NOT NULL DEFAULT 0,
  reps             INTEGER NOT NULL DEFAULT 0,
  lapses           INTEGER NOT NULL DEFAULT 0,
  due_at           INTEGER,
  last_reviewed_at INTEGER,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
-- Load-bearing: acceptance of a review candidate upserts on it, which is what stops
-- two candidates carrying the same skill label from splitting one posterior in two.
CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_subject_name ON skills(subject_id, name);
-- The tick's hot predicate is `status = 'active' AND due_at <= ?`. The
-- subject-leading index below cannot serve it, so without this the due roll is a
-- full table scan every tick, forever.
CREATE INDEX IF NOT EXISTS idx_skills_due ON skills(status, due_at);
-- 'What should I study next' = lowest mastery among a subject's active skills.
CREATE INDEX IF NOT EXISTS idx_skills_subject_mastery
  ON skills(subject_id, status, mastery);

CREATE TABLE IF NOT EXISTS skill_prereqs (
  skill_id   TEXT NOT NULL,
  prereq_id  TEXT NOT NULL,
  -- Denormalized so the subject cascade is a single-table delete. Edges never
  -- cross subjects; `add_prereq` refuses one that would.
  subject_id TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (skill_id, prereq_id)
);
CREATE INDEX IF NOT EXISTS idx_skill_prereqs_prereq ON skill_prereqs(prereq_id);
CREATE INDEX IF NOT EXISTS idx_skill_prereqs_subject ON skill_prereqs(subject_id);

CREATE TABLE IF NOT EXISTS items (
  id           TEXT PRIMARY KEY,
  subject_id   TEXT NOT NULL,
  skill_id     TEXT NOT NULL,
  kind         TEXT NOT NULL,
  prompt       TEXT NOT NULL,
  choices      TEXT,
  -- A tagged JSON `AnswerKey`, so the grader gets a total match and an item that
  -- is missing its answer cannot be constructed.
  answer       TEXT NOT NULL,
  origin       TEXT NOT NULL,
  origin_model TEXT,
  source_id    TEXT,
  source_ref   TEXT,
  version      INTEGER NOT NULL DEFAULT 1,
  archived     INTEGER NOT NULL DEFAULT 0,
  created_at   INTEGER NOT NULL,
  updated_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_skill ON items(skill_id, archived, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_subject ON items(subject_id, archived);

CREATE TABLE IF NOT EXISTS attempts (
  id             TEXT PRIMARY KEY,
  subject_id     TEXT NOT NULL,
  skill_id       TEXT NOT NULL,
  item_id        TEXT NOT NULL,
  session_id     TEXT,
  -- The item version this was graded against, so a rewritten question does not
  -- retroactively change what a past answer meant.
  item_version   INTEGER NOT NULL,
  response       TEXT NOT NULL,
  -- NULL while a free-response answer awaits its rubric mark. That state is what
  -- keeps `mastery.dropped` from firing on an ungraded answer.
  correct        INTEGER,
  score          REAL,
  graded_by      TEXT,
  feedback       TEXT,
  latency_ms     INTEGER,
  mastery_before REAL NOT NULL,
  mastery_after  REAL NOT NULL,
  informative    INTEGER NOT NULL DEFAULT 1,
  at             INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_skill ON attempts(skill_id, at DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_subject ON attempts(subject_id, at DESC);
CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id);
-- Feeds the planner's cost model (median observed latency) and the rubric grader's
-- 'what is still awaiting a mark' scan.
CREATE INDEX IF NOT EXISTS idx_attempts_item ON attempts(item_id, at DESC);

CREATE TABLE IF NOT EXISTS sessions (
  id              TEXT PRIMARY KEY,
  subject_id      TEXT NOT NULL,
  planned_minutes INTEGER NOT NULL,
  status          TEXT NOT NULL,
  -- Written once at finish. The trajectory projection reads the trailing ten of
  -- these on every tick; recomputing them would mean joining every attempt of every
  -- one of those sessions each time, for a number that cannot change again.
  mastery_gain    REAL,
  summary         TEXT,
  created_at      INTEGER NOT NULL,
  started_at      INTEGER,
  finished_at     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sessions_subject ON sessions(subject_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_finished
  ON sessions(subject_id, status, finished_at DESC);

CREATE TABLE IF NOT EXISTS session_items (
  session_id  TEXT NOT NULL,
  position    INTEGER NOT NULL,
  item_id     TEXT NOT NULL,
  skill_id    TEXT NOT NULL,
  -- The planner's own estimates, kept so a finished session can be compared with
  -- what it predicted. The planner is only as good as its cost model and this is
  -- the only record of what that model said.
  est_cost_ms INTEGER NOT NULL,
  est_gain    REAL NOT NULL,
  attempt_id  TEXT,
  PRIMARY KEY (session_id, position)
);
-- One plan must not ask the same question twice.
CREATE UNIQUE INDEX IF NOT EXISTS idx_session_items_item
  ON session_items(session_id, item_id);

CREATE TABLE IF NOT EXISTS review_candidates (
  id              TEXT PRIMARY KEY,
  subject_id      TEXT NOT NULL,
  conversation_id TEXT,
  agent_id        TEXT,
  prompt          TEXT NOT NULL,
  answer          TEXT NOT NULL,
  skill_label     TEXT,
  status          TEXT NOT NULL,
  -- The hook's KV key and this candidate's offset within its payload. See the
  -- module docs: this pair is what makes a re-delivered drain a no-op.
  source_key      TEXT NOT NULL,
  source_index    INTEGER NOT NULL,
  item_id         TEXT,
  created_at      INTEGER NOT NULL,
  decided_at      INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_review_candidates_source
  ON review_candidates(source_key, source_index);
CREATE INDEX IF NOT EXISTS idx_review_candidates_queue
  ON review_candidates(subject_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS event_marks (
  kind            TEXT NOT NULL,
  subject_id      TEXT NOT NULL,
  -- '' for the subject-scoped events (`goal.at-risk`), a skill id for the
  -- per-skill ones. NOT NULL with a '' default on purpose: SQLite does not treat
  -- two NULLs as equal, so a nullable column here would let the primary key admit
  -- unlimited duplicate marks and the cooldown would silently do nothing.
  ref_id          TEXT NOT NULL DEFAULT '',
  last_emitted_at INTEGER NOT NULL,
  PRIMARY KEY (kind, subject_id, ref_id)
);

CREATE TABLE IF NOT EXISTS settings (
  scope      TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
";

// ── Column lists ───────────────────────────────────────────────────────────────
//
// Shared by every SELECT and its `row_to_*` decoder, so the two cannot drift.
// Never `SELECT *`: the column ORDER of a `*` is whatever the table happens to
// have, which makes adding a column a silent off-by-one in every decoder.

const COLS_SUBJECT: &str = "id, name, detail, exam_date, timezone, created_at, updated_at";
const COLS_SOURCE: &str =
    "id, subject_id, kind, title, uri, parsed_text, parser, created_at";
const COLS_SKILL: &str = "id, subject_id, name, detail, status, source_id, p_init, p_transit, \
                          p_slip, p_guess, mastery, ease, interval_days, reps, lapses, due_at, \
                          last_reviewed_at, created_at, updated_at";
const COLS_ITEM: &str = "id, subject_id, skill_id, kind, prompt, choices, answer, origin, \
                         origin_model, source_id, source_ref, version, archived, created_at, \
                         updated_at";
const COLS_ATTEMPT: &str = "id, subject_id, skill_id, item_id, session_id, item_version, \
                            response, correct, score, graded_by, feedback, latency_ms, \
                            mastery_before, mastery_after, informative, at";
const COLS_SESSION: &str = "id, subject_id, planned_minutes, status, mastery_gain, summary, \
                            created_at, started_at, finished_at";
const COLS_SESSION_ITEM: &str =
    "session_id, position, item_id, skill_id, est_cost_ms, est_gain, attempt_id";
const COLS_CANDIDATE: &str = "id, subject_id, conversation_id, agent_id, prompt, answer, \
                              skill_label, status, source_key, source_index, item_id, \
                              created_at, decided_at";

// ── Subjects ───────────────────────────────────────────────────────────────────

impl TuitionStore {
    pub async fn list_subjects(&self) -> Result<Vec<Subject>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_SUBJECT} FROM subjects ORDER BY created_at ASC, id ASC");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], row_to_subject)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn get_subject(&self, id: &str) -> Result<Option<Subject>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_SUBJECT} FROM subjects WHERE id = ?1");
        Ok(conn
            .query_row(&sql, params![id], row_to_subject)
            .optional()?)
    }

    pub async fn create_subject(
        &self,
        name: &str,
        detail: Option<&str>,
        exam_date: Option<&str>,
        timezone: Option<&str>,
    ) -> Result<Subject> {
        let now = now_ms();
        let subject = Subject {
            id: new_id(ID_SUBJECT),
            name: name.trim().to_string(),
            detail: detail.map(|d| d.trim().to_string()).filter(|d| !d.is_empty()),
            exam_date: exam_date
                .map(|d| d.trim().to_string())
                .filter(|d| !d.is_empty()),
            timezone: timezone
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .unwrap_or(DEFAULT_TIMEZONE)
                .to_string(),
            created_at: now,
            updated_at: now,
        };
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO subjects (id, name, detail, exam_date, timezone, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                subject.id,
                subject.name,
                subject.detail,
                subject.exam_date,
                subject.timezone,
                subject.created_at,
                subject.updated_at
            ],
        )?;
        Ok(subject)
    }

    /// Full-field update. Returns `false` when no row matched, so the caller can
    /// 404 instead of reporting a successful no-op.
    pub async fn update_subject(
        &self,
        id: &str,
        name: &str,
        detail: Option<&str>,
        exam_date: Option<&str>,
        timezone: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE subjects SET name = ?2, detail = ?3, exam_date = ?4, timezone = ?5,
                    updated_at = ?6
             WHERE id = ?1",
            params![
                id,
                name.trim(),
                detail.map(str::trim).filter(|d| !d.is_empty()),
                exam_date.map(str::trim).filter(|d| !d.is_empty()),
                if timezone.trim().is_empty() {
                    DEFAULT_TIMEZONE
                } else {
                    timezone.trim()
                },
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Delete a subject and everything under it.
    ///
    /// An explicit ordered cascade in ONE transaction (see the module docs for why
    /// not `ON DELETE CASCADE`). The order is load-bearing in two places:
    /// `session_items` is reached through `sessions` and must go first, and
    /// `skill_prereqs` is reached through `skills` and must go before them. Both
    /// carry a denormalized `subject_id` precisely so those deletes stay
    /// single-table and cannot be broken by a later index change.
    pub async fn delete_subject(&self, id: &str) -> Result<bool> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM attempts WHERE subject_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM session_items WHERE session_id IN
                 (SELECT id FROM sessions WHERE subject_id = ?1)",
            params![id],
        )?;
        tx.execute("DELETE FROM sessions WHERE subject_id = ?1", params![id])?;
        tx.execute("DELETE FROM items WHERE subject_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM skill_prereqs WHERE subject_id = ?1",
            params![id],
        )?;
        tx.execute("DELETE FROM skills WHERE subject_id = ?1", params![id])?;
        tx.execute("DELETE FROM sources WHERE subject_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM review_candidates WHERE subject_id = ?1",
            params![id],
        )?;
        tx.execute("DELETE FROM event_marks WHERE subject_id = ?1", params![id])?;
        let n = tx.execute("DELETE FROM subjects WHERE id = ?1", params![id])?;
        tx.commit()?;
        Ok(n > 0)
    }

    /// The `tuition_subjects` data category: every subject on this node, with
    /// everything under it.
    ///
    /// Not a loop over [`Self::delete_subject`], because a wipe that fails halfway
    /// through leaves a half-deleted node. One transaction, same order, so it is
    /// all or nothing. Settings survive deliberately — they are configuration, not
    /// study material, and the confirm dialog promises the material.
    pub async fn purge_all(&self) -> Result<()> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        for table in [
            "attempts",
            "session_items",
            "sessions",
            "items",
            "skill_prereqs",
            "skills",
            "sources",
            "review_candidates",
            "event_marks",
            "subjects",
        ] {
            tx.execute(&format!("DELETE FROM {table}"), [])?;
        }
        tx.commit()?;
        Ok(())
    }
}

// ── Sources ────────────────────────────────────────────────────────────────────

impl TuitionStore {
    pub async fn list_sources(&self, subject_id: &str) -> Result<Vec<Source>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_SOURCE} FROM sources WHERE subject_id = ?1
             ORDER BY created_at DESC, id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![subject_id], row_to_source)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn get_source(&self, id: &str) -> Result<Option<Source>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_SOURCE} FROM sources WHERE id = ?1");
        Ok(conn.query_row(&sql, params![id], row_to_source).optional()?)
    }

    pub async fn create_source(
        &self,
        subject_id: &str,
        kind: SourceKind,
        title: &str,
        uri: Option<&str>,
        parsed_text: &str,
        parser: Option<&str>,
    ) -> Result<Source> {
        let source = Source {
            id: new_id(ID_SOURCE),
            subject_id: subject_id.to_string(),
            kind,
            title: title.trim().to_string(),
            uri: uri.map(str::to_string),
            parsed_text: parsed_text.to_string(),
            parser: parser.map(str::to_string),
            created_at: now_ms(),
        };
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO sources (id, subject_id, kind, title, uri, parsed_text, parser, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                source.id,
                source.subject_id,
                source.kind.as_str(),
                source.title,
                source.uri,
                source.parsed_text,
                source.parser,
                source.created_at
            ],
        )?;
        Ok(source)
    }

    /// Delete a source. Skills and items keep their `source_id` and it is allowed
    /// to dangle: the provenance of an item ("this came from chapter 4") remains
    /// true after the chapter is removed from the app, and blanking it would erase
    /// the answer to "where did this question come from".
    pub async fn delete_source(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute("DELETE FROM sources WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }
}

// ── Skills ─────────────────────────────────────────────────────────────────────

impl TuitionStore {
    pub async fn list_skills(
        &self,
        subject_id: &str,
        status: Option<SkillStatus>,
    ) -> Result<Vec<Skill>> {
        let conn = self.conn.lock().await;
        // Ordered by mastery then id: this is the "weakest first" list the study
        // planner and the mastery report both read, and the id tiebreak is what
        // makes the same data produce the same order on every run.
        let sql = format!(
            "SELECT {COLS_SKILL} FROM skills
             WHERE subject_id = ?1 AND (?2 IS NULL OR status = ?2)
             ORDER BY mastery ASC, id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![subject_id, status.map(SkillStatus::as_str)],
            row_to_skill,
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn get_skill(&self, id: &str) -> Result<Option<Skill>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_SKILL} FROM skills WHERE id = ?1");
        Ok(conn.query_row(&sql, params![id], row_to_skill).optional()?)
    }

    /// Create a skill, or fold the offer into the existing one with the same name
    /// in the same subject.
    ///
    /// An upsert rather than an insert because both writers of skills — ingest
    /// proposing a batch, and the acceptance of a Study-mode candidate whose short
    /// label repeats across a topic — legitimately re-offer a name. `DO UPDATE`
    /// rather than `DO NOTHING` so `RETURNING` always yields the row; `DO NOTHING`
    /// returns nothing on conflict and the caller would have to re-SELECT.
    ///
    /// **What a conflict does and does not overwrite** matters, because both
    /// callers get this wrong in opposite directions:
    ///
    /// - The *learned* half — `mastery`, `ease`, the interval, the reps, the BKT
    ///   parameters — is never touched. That is what makes re-offering a name safe:
    ///   a second ingest of the same chapter must not reset what the learner knows.
    /// - `status` only ever moves FORWARD out of `proposed`. Accepting a candidate
    ///   whose label matches a still-proposed skill therefore activates it — without
    ///   this, acceptance would create an item under a skill that
    ///   [`Self::list_due_skills`] filters out, and the learner would accept
    ///   candidates that never came up for review. A re-proposal from ingest cannot
    ///   demote an active skill back, and an archived one is not silently revived.
    /// - `detail` and `source_id` fill in only when they were empty, so the later,
    ///   thinner offer cannot blank what the first one recorded.
    pub async fn upsert_skill(
        &self,
        subject_id: &str,
        name: &str,
        detail: Option<&str>,
        status: SkillStatus,
        source_id: Option<&str>,
        params_in: BktParams,
    ) -> Result<Skill> {
        let bkt = params_in.clamped();
        let now = now_ms();
        let conn = self.conn.lock().await;
        let sql = format!(
            "INSERT INTO skills (id, subject_id, name, detail, status, source_id, p_init,
                                 p_transit, p_slip, p_guess, mastery, ease, interval_days,
                                 reps, lapses, due_at, last_reviewed_at, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, 0, 0, 0, NULL, NULL,
                     ?13, ?13)
             ON CONFLICT(subject_id, name) DO UPDATE SET
                 detail = COALESCE(skills.detail, excluded.detail),
                 source_id = COALESCE(skills.source_id, excluded.source_id),
                 status = CASE WHEN skills.status = 'proposed'
                               THEN excluded.status ELSE skills.status END,
                 updated_at = excluded.updated_at
             RETURNING {COLS_SKILL}"
        );
        let skill = conn.query_row(
            &sql,
            params![
                new_id(ID_SKILL),
                subject_id,
                name.trim(),
                detail.map(str::trim).filter(|d| !d.is_empty()),
                status.as_str(),
                source_id,
                bkt.p_init,
                bkt.p_transit,
                bkt.p_slip,
                bkt.p_guess,
                // Seeded to `p_init`: the posterior before any evidence IS the prior.
                bkt.p_init,
                DEFAULT_EASE,
                now
            ],
            row_to_skill,
        )?;
        Ok(skill)
    }

    /// Edit the authored half of a skill (name, detail, status, BKT parameters).
    /// The learned half — mastery and the schedule — is only ever moved by
    /// [`Self::update_skill_mastery`] and [`Self::update_skill_schedule`], so a
    /// rename cannot silently reset what the learner knows.
    pub async fn update_skill(
        &self,
        id: &str,
        name: &str,
        detail: Option<&str>,
        status: SkillStatus,
        params_in: BktParams,
    ) -> Result<bool> {
        let bkt = params_in.clamped();
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE skills SET name = ?2, detail = ?3, status = ?4, p_init = ?5, p_transit = ?6,
                    p_slip = ?7, p_guess = ?8, updated_at = ?9
             WHERE id = ?1",
            params![
                id,
                name.trim(),
                detail.map(str::trim).filter(|d| !d.is_empty()),
                status.as_str(),
                bkt.p_init,
                bkt.p_transit,
                bkt.p_slip,
                bkt.p_guess,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Write the posterior a BKT update produced. Clamped into `[0,1]` here as well
    /// as in the update itself, because this is the last gate before it becomes the
    /// prior of every future attempt.
    pub async fn update_skill_mastery(&self, id: &str, mastery: f64) -> Result<bool> {
        let mastery = if mastery.is_finite() {
            mastery.clamp(0.0, 1.0)
        } else {
            return Err(anyhow::anyhow!(
                "refusing to store a non-finite mastery for skill {id}"
            ));
        };
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE skills SET mastery = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, mastery, now_ms()],
        )?;
        Ok(n > 0)
    }

    /// Write the SM-2 state a session's grade produced. `due_at` is expected to be
    /// a day boundary in the subject's timezone — computed by the caller through
    /// [`crate::models::day_start_plus_days`], not here, because the store has no
    /// business reading a clock.
    pub async fn update_skill_schedule(
        &self,
        id: &str,
        ease: f64,
        interval_days: u32,
        reps: u32,
        lapses: u32,
        due_at: i64,
        reviewed_at: i64,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE skills SET ease = ?2, interval_days = ?3, reps = ?4, lapses = ?5,
                    due_at = ?6, last_reviewed_at = ?7, updated_at = ?8
             WHERE id = ?1",
            params![
                id,
                ease.max(MIN_EASE),
                interval_days,
                reps,
                lapses,
                due_at,
                reviewed_at,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Delete a skill and everything derived from it, in one transaction.
    /// Prerequisite edges are removed from BOTH directions — an edge naming a
    /// skill that no longer exists would make the graph walk chase an id that
    /// resolves to nothing, and `ready_skills` would then treat it as unmastered
    /// forever.
    pub async fn delete_skill(&self, id: &str) -> Result<bool> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM attempts WHERE skill_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM session_items WHERE skill_id = ?1",
            params![id],
        )?;
        tx.execute("DELETE FROM items WHERE skill_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM skill_prereqs WHERE skill_id = ?1 OR prereq_id = ?1",
            params![id],
        )?;
        let n = tx.execute("DELETE FROM skills WHERE id = ?1", params![id])?;
        tx.commit()?;
        Ok(n > 0)
    }

    /// Skills whose review is owed as of `now`.
    ///
    /// A never-reviewed skill (`due_at IS NULL`) is deliberately NOT due: it has
    /// never been scheduled, so it is new work rather than an overdue review, and
    /// `review.due` firing for it would mean every freshly ingested syllabus
    /// notifies the learner about a hundred skills at once.
    pub async fn list_due_skills(
        &self,
        subject_id: Option<&str>,
        now: i64,
        limit: i64,
    ) -> Result<Vec<Skill>> {
        let conn = self.conn.lock().await;
        // `due_at IS NULL` means "never reviewed", and a never-reviewed skill is the
        // most due thing there is — it is brand-new material the learner has not seen.
        // Requiring a non-null `due_at` here made a freshly created skill permanently
        // unstudyable: nothing was ever planned for it, so it never got an attempt, so
        // it never got a due date. The whole app was dead on arrival for new material
        // and every unit test passed, because they all seeded a due date.
        //
        // Overdue REVIEWS sort before new material (`due_at IS NULL` last), which is
        // the standard spaced-repetition ordering: retention of what you have already
        // learned is worth more than volume of what you have not.
        let sql = format!(
            "SELECT {COLS_SKILL} FROM skills
             WHERE status = 'active' AND (due_at IS NULL OR due_at <= ?1)
               AND (?2 IS NULL OR subject_id = ?2)
             ORDER BY (due_at IS NULL) ASC, due_at ASC, id ASC LIMIT ?3"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![now, subject_id, clamp_limit(limit)], row_to_skill)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
}

// ── Prerequisite graph ─────────────────────────────────────────────────────────

impl TuitionStore {
    /// The direct prerequisites of one skill.
    pub async fn list_prereqs(&self, skill_id: &str) -> Result<Vec<String>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT prereq_id FROM skill_prereqs WHERE skill_id = ?1 ORDER BY prereq_id ASC",
        )?;
        let rows = stmt.query_map(params![skill_id], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Every edge in a subject, in a stable order. This is what the topological
    /// "what should I study next" reads, and an unstable order there would produce
    /// a different answer on a replay of identical data.
    pub async fn list_prereq_edges(&self, subject_id: &str) -> Result<Vec<PrereqEdge>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT skill_id, prereq_id FROM skill_prereqs WHERE subject_id = ?1
             ORDER BY skill_id ASC, prereq_id ASC",
        )?;
        let rows = stmt.query_map(params![subject_id], |r| {
            Ok(PrereqEdge {
                skill_id: r.get(0)?,
                prereq_id: r.get(1)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Add `prereq_id` as a prerequisite of `skill_id`.
    ///
    /// **Cycles are rejected here, at write time, with the offending path named.**
    /// There is no good runtime recovery from one: "what should I study next" walks
    /// prerequisites and a cycle makes it non-terminating, so the only alternatives
    /// to refusing the write are a graph that hangs the app or one that silently
    /// drops an edge the learner thinks they added.
    ///
    /// The check and the insert share one transaction, so two concurrent writes
    /// cannot each observe an acyclic graph and jointly close a cycle.
    pub async fn add_prereq(&self, skill_id: &str, prereq_id: &str) -> Result<()> {
        if skill_id == prereq_id {
            anyhow::bail!("a skill cannot be its own prerequisite");
        }
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;

        let subject_of = |id: &str| -> Result<String> {
            tx.query_row(
                "SELECT subject_id FROM skills WHERE id = ?1",
                params![id],
                |r| r.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("skill {id} does not exist"))
        };
        let subject = subject_of(skill_id)?;
        if subject_of(prereq_id)? != subject {
            // Edges are allowed to be denormalized onto one `subject_id` only
            // because this holds; the delete cascade depends on it.
            anyhow::bail!("a prerequisite must belong to the same subject");
        }

        // The edge means "prereq_id must be mastered before skill_id". Adding it
        // closes a cycle exactly when `skill_id` is ALREADY a transitive
        // prerequisite of `prereq_id` — so walk up from `prereq_id` and see whether
        // we arrive back at `skill_id`.
        if let Some(path) = prereq_path_to(&tx, prereq_id, skill_id)? {
            let named = name_path(&tx, &path)?;
            anyhow::bail!(
                "adding this prerequisite would close a cycle: {}",
                named.join(" → ")
            );
        }

        tx.execute(
            "INSERT OR IGNORE INTO skill_prereqs (skill_id, prereq_id, subject_id, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![skill_id, prereq_id, subject, now_ms()],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub async fn remove_prereq(&self, skill_id: &str, prereq_id: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "DELETE FROM skill_prereqs WHERE skill_id = ?1 AND prereq_id = ?2",
            params![skill_id, prereq_id],
        )?;
        Ok(n > 0)
    }
}

/// Walk the prerequisite closure upward from `from`, looking for `target`.
///
/// Returns the path `from → … → target` when one exists. Carries a visited set and
/// a hard expansion cap: without them a graph that is already cyclic (an older
/// build, a `sqlite3` shell edit) would loop forever *while holding the connection
/// mutex*, hanging every request in the process including `/health`.
fn prereq_path_to(
    conn: &Connection,
    from: &str,
    target: &str,
) -> Result<Option<Vec<String>>> {
    let mut stmt = conn.prepare("SELECT prereq_id FROM skill_prereqs WHERE skill_id = ?1")?;
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut parent: HashMap<String, String> = HashMap::new();
    let mut frontier = vec![from.to_string()];
    visited.insert(from.to_string());
    let mut expansions = 0usize;

    while let Some(current) = frontier.pop() {
        expansions += 1;
        if expansions > MAX_PREREQ_WALK {
            anyhow::bail!(
                "the prerequisite graph is too large or already cyclic to check safely"
            );
        }
        let next: Vec<String> = stmt
            .query_map(params![current], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for id in next {
            if !visited.insert(id.clone()) {
                continue;
            }
            parent.insert(id.clone(), current.clone());
            if id == target {
                // Reconstruct from the target back to the origin, then flip it.
                let mut path = vec![id];
                while let Some(prev) = parent.get(path.last().expect("non-empty")) {
                    path.push(prev.clone());
                    if prev == from {
                        break;
                    }
                }
                path.reverse();
                return Ok(Some(path));
            }
            frontier.push(id);
        }
    }
    Ok(None)
}

/// Skill names for a path of ids, falling back to the id when the row is gone —
/// an error message is not worth failing a write-rejection over.
fn name_path(conn: &Connection, ids: &[String]) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT name FROM skills WHERE id = ?1")?;
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let name: Option<String> = stmt
            .query_row(params![id], |r| r.get::<_, String>(0))
            .optional()?;
        out.push(name.unwrap_or_else(|| id.clone()));
    }
    Ok(out)
}

// ── Items ──────────────────────────────────────────────────────────────────────

impl TuitionStore {
    pub async fn list_items(
        &self,
        skill_id: &str,
        include_archived: bool,
        limit: i64,
    ) -> Result<Vec<Item>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_ITEM} FROM items
             WHERE skill_id = ?1 AND (?2 = 1 OR archived = 0)
             ORDER BY created_at ASC, id ASC LIMIT ?3"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![skill_id, include_archived, clamp_limit(limit)],
            row_to_item,
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn list_items_for_subject(
        &self,
        subject_id: &str,
        include_archived: bool,
        limit: i64,
    ) -> Result<Vec<Item>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_ITEM} FROM items
             WHERE subject_id = ?1 AND (?2 = 1 OR archived = 0)
             ORDER BY created_at ASC, id ASC LIMIT ?3"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![subject_id, include_archived, clamp_limit(limit)],
            row_to_item,
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn get_item(&self, id: &str) -> Result<Option<Item>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_ITEM} FROM items WHERE id = ?1");
        Ok(conn.query_row(&sql, params![id], row_to_item).optional()?)
    }

    /// Insert a practice item.
    ///
    /// Two invariants are enforced here rather than trusted from the caller: the
    /// answer key must grade the declared kind (an `mcq` item carrying a numeric
    /// key is ungradeable in a way nothing detects until the learner is looking at
    /// it), and `subject_id` is read off the skill instead of being supplied.
    pub async fn create_item(&self, new: &NewItem) -> Result<Item> {
        if new.answer.kind() != Some(new.kind) {
            anyhow::bail!(
                "this item is declared `{}` but its answer key grades a different kind",
                new.kind.as_str()
            );
        }
        let answer = new.answer.encode()?;
        let choices = if new.choices.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&new.choices)?)
        };
        let conn = self.conn.lock().await;
        let subject_id: String = conn
            .query_row(
                "SELECT subject_id FROM skills WHERE id = ?1",
                params![new.skill_id],
                |r| r.get(0),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("skill {} does not exist", new.skill_id))?;
        let now = now_ms();
        let sql = format!(
            "INSERT INTO items (id, subject_id, skill_id, kind, prompt, choices, answer, origin,
                                origin_model, source_id, source_ref, version, archived,
                                created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 1, 0, ?12, ?12)
             RETURNING {COLS_ITEM}"
        );
        let item = conn.query_row(
            &sql,
            params![
                new_id(ID_ITEM),
                subject_id,
                new.skill_id,
                new.kind.as_str(),
                new.prompt.trim(),
                choices,
                answer,
                new.origin.as_str(),
                new.origin_model,
                new.source_id,
                new.source_ref,
                now
            ],
            row_to_item,
        )?;
        Ok(item)
    }

    /// Edit an item's question and answer, bumping its `version`.
    ///
    /// The bump is the point: attempts record the version they were graded under,
    /// so rewriting a question does not retroactively change what a past answer
    /// meant. The kind is NOT editable — changing it would silently reinterpret
    /// every stored attempt against a different grader.
    pub async fn update_item(
        &self,
        id: &str,
        prompt: &str,
        choices: &[Choice],
        answer: &AnswerKey,
        source_ref: Option<&str>,
    ) -> Result<bool> {
        let encoded = answer.encode()?;
        let choices_json = if choices.is_empty() {
            None
        } else {
            Some(serde_json::to_string(choices)?)
        };
        let conn = self.conn.lock().await;
        let kind: Option<String> = conn
            .query_row("SELECT kind FROM items WHERE id = ?1", params![id], |r| {
                r.get(0)
            })
            .optional()?;
        let Some(kind) = kind else {
            return Ok(false);
        };
        if answer.kind() != Some(ItemKind::parse(&kind)) {
            anyhow::bail!("an item's kind cannot change: this answer key grades a different one");
        }
        let n = conn.execute(
            "UPDATE items SET prompt = ?2, choices = ?3, answer = ?4, source_ref = ?5,
                    version = version + 1, updated_at = ?6
             WHERE id = ?1",
            params![
                id,
                prompt.trim(),
                choices_json,
                encoded,
                source_ref,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Archive rather than delete is the normal retirement: an item's attempts are
    /// evidence behind a posterior, and deleting the question they answered leaves
    /// the number with nothing behind it.
    pub async fn set_item_archived(&self, id: &str, archived: bool) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE items SET archived = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, archived, now_ms()],
        )?;
        Ok(n > 0)
    }

    /// Hard delete, with the attempts that cite it. Offered alongside archiving for
    /// the case archiving does not cover: a generated item that is simply wrong,
    /// whose attempts are evidence of nothing.
    pub async fn delete_item(&self, id: &str) -> Result<bool> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM attempts WHERE item_id = ?1", params![id])?;
        tx.execute("DELETE FROM session_items WHERE item_id = ?1", params![id])?;
        let n = tx.execute("DELETE FROM items WHERE id = ?1", params![id])?;
        tx.commit()?;
        Ok(n > 0)
    }
}

// ── Attempts ───────────────────────────────────────────────────────────────────

impl TuitionStore {
    /// Record an answer.
    ///
    /// `subject_id` and `item_version` are read off the item row rather than taken
    /// from the caller — they are denormalized copies with no foreign key behind
    /// them, so the only way they stay true is if exactly one place writes them.
    /// The skill is checked against the item's for the same reason: an attempt
    /// filed under the wrong skill moves the wrong posterior, and nothing
    /// downstream could ever detect it.
    pub async fn record_attempt(&self, new: &NewAttempt) -> Result<Attempt> {
        let conn = self.conn.lock().await;
        let (subject_id, skill_id, version): (String, String, i64) = conn
            .query_row(
                "SELECT subject_id, skill_id, version FROM items WHERE id = ?1",
                params![new.item_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?
            .ok_or_else(|| anyhow::anyhow!("item {} does not exist", new.item_id))?;
        if skill_id != new.skill_id {
            anyhow::bail!("item {} does not belong to skill {}", new.item_id, new.skill_id);
        }
        let sql = format!(
            "INSERT INTO attempts (id, subject_id, skill_id, item_id, session_id, item_version,
                                   response, correct, score, graded_by, feedback, latency_ms,
                                   mastery_before, mastery_after, informative, at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
             RETURNING {COLS_ATTEMPT}"
        );
        let attempt = conn.query_row(
            &sql,
            params![
                new_id(ID_ATTEMPT),
                subject_id,
                skill_id,
                new.item_id,
                new.session_id,
                version,
                new.response,
                new.correct,
                new.score,
                new.graded_by.map(GradedBy::as_str),
                new.feedback,
                new.latency_ms,
                new.mastery_before,
                new.mastery_after,
                new.informative,
                now_ms()
            ],
            row_to_attempt,
        )?;
        Ok(attempt)
    }

    /// Land a rubric mark on a free-response attempt that was recorded ungraded.
    ///
    /// A compare-and-swap on `correct IS NULL`: the grade may only be written once,
    /// so a retried grading call (or two graders racing) cannot move the posterior
    /// twice. `false` means someone already graded it.
    pub async fn grade_attempt(
        &self,
        id: &str,
        correct: bool,
        score: Option<f64>,
        graded_by: GradedBy,
        feedback: Option<&str>,
        mastery_after: f64,
        informative: bool,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE attempts SET correct = ?2, score = ?3, graded_by = ?4, feedback = ?5,
                    mastery_after = ?6, informative = ?7
             WHERE id = ?1 AND correct IS NULL",
            params![
                id,
                correct,
                score,
                graded_by.as_str(),
                feedback,
                mastery_after,
                informative
            ],
        )?;
        Ok(n > 0)
    }

    pub async fn get_attempt(&self, id: &str) -> Result<Option<Attempt>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_ATTEMPT} FROM attempts WHERE id = ?1");
        Ok(conn
            .query_row(&sql, params![id], row_to_attempt)
            .optional()?)
    }

    pub async fn list_attempts_for_skill(&self, skill_id: &str, limit: i64) -> Result<Vec<Attempt>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_ATTEMPT} FROM attempts WHERE skill_id = ?1
             ORDER BY at DESC, id DESC LIMIT ?2"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![skill_id, clamp_limit(limit)], row_to_attempt)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn list_attempts_for_subject(
        &self,
        subject_id: &str,
        limit: i64,
    ) -> Result<Vec<Attempt>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_ATTEMPT} FROM attempts WHERE subject_id = ?1
             ORDER BY at DESC, id DESC LIMIT ?2"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![subject_id, clamp_limit(limit)], row_to_attempt)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn list_attempts_for_session(&self, session_id: &str) -> Result<Vec<Attempt>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_ATTEMPT} FROM attempts WHERE session_id = ?1 ORDER BY at ASC, id ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![session_id], row_to_attempt)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Observed answer latencies for one item kind within a subject, ascending.
    ///
    /// Returned as a list rather than a median, because the planner's cost model is
    /// the planner's to define — the store's job is to hand it the observations,
    /// and `MEDIAN()` is not a SQLite function anyway.
    pub async fn latencies_for_kind(
        &self,
        subject_id: &str,
        kind: ItemKind,
        limit: i64,
    ) -> Result<Vec<i64>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT a.latency_ms FROM attempts a
             JOIN items i ON i.id = a.item_id
             WHERE a.subject_id = ?1 AND i.kind = ?2 AND a.latency_ms IS NOT NULL
             ORDER BY a.latency_ms ASC LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            params![subject_id, kind.as_str(), clamp_limit(limit)],
            |r| r.get::<_, i64>(0),
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
}

// ── Sessions ───────────────────────────────────────────────────────────────────

impl TuitionStore {
    /// Persist a plan before its first question is shown.
    ///
    /// Written up front, not accumulated as the sitting goes: the first answer
    /// moves the schedule the planner selected against, so re-planning on a reload
    /// would produce a different sitting from the one the learner started.
    pub async fn create_session(
        &self,
        subject_id: &str,
        planned_minutes: u32,
        plan: &[PlannedItem],
    ) -> Result<Session> {
        let now = now_ms();
        let session = Session {
            id: new_id(ID_SESSION),
            subject_id: subject_id.to_string(),
            planned_minutes,
            status: SessionStatus::Planned,
            mastery_gain: None,
            summary: None,
            created_at: now,
            started_at: None,
            finished_at: None,
        };
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO sessions (id, subject_id, planned_minutes, status, mastery_gain,
                                   summary, created_at, started_at, finished_at)
             VALUES (?1, ?2, ?3, ?4, NULL, NULL, ?5, NULL, NULL)",
            params![
                session.id,
                session.subject_id,
                session.planned_minutes,
                session.status.as_str(),
                session.created_at
            ],
        )?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO session_items (session_id, position, item_id, skill_id,
                                            est_cost_ms, est_gain, attempt_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL)",
            )?;
            // Positions come from the slice order, so a plan cannot arrive with two
            // items claiming the same position.
            for (position, planned) in plan.iter().enumerate() {
                stmt.execute(params![
                    session.id,
                    position as i64,
                    planned.item_id,
                    planned.skill_id,
                    planned.est_cost_ms,
                    planned.est_gain
                ])?;
            }
        }
        tx.commit()?;
        Ok(session)
    }

    pub async fn get_session(&self, id: &str) -> Result<Option<Session>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_SESSION} FROM sessions WHERE id = ?1");
        Ok(conn
            .query_row(&sql, params![id], row_to_session)
            .optional()?)
    }

    pub async fn list_sessions(&self, subject_id: &str, limit: i64) -> Result<Vec<Session>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_SESSION} FROM sessions WHERE subject_id = ?1
             ORDER BY created_at DESC, id DESC LIMIT ?2"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![subject_id, clamp_limit(limit)], row_to_session)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// The trailing finished sessions the trajectory projection averages over,
    /// most recent first. Only sessions that actually recorded a gain count — an
    /// abandoned sitting is not evidence of a learning rate.
    pub async fn list_recent_finished_sessions(
        &self,
        subject_id: &str,
        limit: i64,
    ) -> Result<Vec<Session>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_SESSION} FROM sessions
             WHERE subject_id = ?1 AND status = 'finished' AND mastery_gain IS NOT NULL
             ORDER BY finished_at DESC, id DESC LIMIT ?2"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![subject_id, clamp_limit(limit)], row_to_session)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn list_session_items(&self, session_id: &str) -> Result<Vec<SessionItem>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_SESSION_ITEM} FROM session_items WHERE session_id = ?1
             ORDER BY position ASC"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![session_id], row_to_session_item)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Move a session to `active`. A CAS on `planned`, so the second caller of a
    /// double-clicked Start does not reset `started_at` and lose the elapsed time.
    pub async fn start_session(&self, id: &str, at: i64) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE sessions SET status = 'active', started_at = ?2
             WHERE id = ?1 AND status = 'planned'",
            params![id, at],
        )?;
        Ok(n > 0)
    }

    /// The next unanswered item on the plan, or `None` when the sitting is done.
    pub async fn next_session_item(&self, session_id: &str) -> Result<Option<SessionItem>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_SESSION_ITEM} FROM session_items
             WHERE session_id = ?1 AND attempt_id IS NULL
             ORDER BY position ASC LIMIT 1"
        );
        Ok(conn
            .query_row(&sql, params![session_id], row_to_session_item)
            .optional()?)
    }

    /// Bind an answer to its slot on the plan.
    ///
    /// A CAS on `attempt_id IS NULL`, so a resubmitted answer cannot displace the
    /// one already on the plan. Note what this does and does not protect: it claims
    /// the SLOT, not the posterior — nothing here stops a caller from applying a
    /// mastery update for an answer that lost this race. That is why the module
    /// docs make this the gate, and the posterior updates conditional on it
    /// returning `true`.
    pub async fn bind_session_attempt(
        &self,
        session_id: &str,
        item_id: &str,
        attempt_id: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE session_items SET attempt_id = ?3
             WHERE session_id = ?1 AND item_id = ?2 AND attempt_id IS NULL",
            params![session_id, item_id, attempt_id],
        )?;
        Ok(n > 0)
    }

    /// Close a session, writing the gain the trajectory projection will read.
    ///
    /// CAS on the two open states so a finished session cannot be finished again
    /// with a different gain — the projection averages these, and a double-count
    /// would inflate the observed learning rate exactly when a learner is checking
    /// whether they are on track.
    pub async fn finish_session(
        &self,
        id: &str,
        status: SessionStatus,
        mastery_gain: Option<f64>,
        summary: Option<&str>,
        at: i64,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE sessions SET status = ?2, mastery_gain = ?3, summary = ?4, finished_at = ?5
             WHERE id = ?1 AND status IN ('planned', 'active')",
            params![id, status.as_str(), mastery_gain, summary, at],
        )?;
        Ok(n > 0)
    }
}

// ── Review candidates ──────────────────────────────────────────────────────────

impl TuitionStore {
    /// File one drained KV payload as pending candidates.
    ///
    /// Returns how many rows were actually inserted. `INSERT OR IGNORE` against the
    /// `(source_key, source_index)` unique index is what makes a re-delivered
    /// payload a no-op: the drain is `keys` → `get` → `delete`, so a crash between
    /// the `get` and the `delete` replays the whole key on the next tick, and a
    /// pre-insert SELECT would still race with a concurrent drain.
    ///
    /// The subject is checked to exist, which the other creates do not bother with:
    /// this `subject_id` originates OUTSIDE the process, in the free-text
    /// `tuition-active-subject` preference. The hook sanitizes its character set
    /// and nothing else, so a typo there would otherwise file candidates under a
    /// subject that no list query names and that `delete_subject` can never reach.
    pub async fn enqueue_candidates(
        &self,
        source_key: &str,
        envelope: &CandidateEnvelope,
    ) -> Result<usize> {
        if envelope.v != CANDIDATE_ENVELOPE_VERSION {
            anyhow::bail!(
                "unrecognized candidate payload version {} under key {source_key}",
                envelope.v
            );
        }
        let now = now_ms();
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let known: Option<i64> = tx
            .query_row(
                "SELECT 1 FROM subjects WHERE id = ?1",
                params![envelope.subject_id],
                |r| r.get(0),
            )
            .optional()?;
        if known.is_none() {
            anyhow::bail!(
                "subject {} does not exist; check the active-subject preference",
                envelope.subject_id
            );
        }
        let mut inserted = 0usize;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO review_candidates
                     (id, subject_id, conversation_id, agent_id, prompt, answer, skill_label,
                      status, source_key, source_index, item_id, created_at, decided_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', ?8, ?9, NULL, ?10, NULL)",
            )?;
            for (index, candidate) in envelope.candidates.iter().enumerate() {
                inserted += stmt.execute(params![
                    new_id(ID_CANDIDATE),
                    envelope.subject_id,
                    envelope.conversation_id,
                    envelope.agent_id,
                    candidate.prompt,
                    candidate.answer,
                    candidate.skill,
                    source_key,
                    index as i64,
                    now
                ])?;
            }
        }
        tx.commit()?;
        Ok(inserted)
    }

    pub async fn list_candidates(
        &self,
        subject_id: Option<&str>,
        status: Option<CandidateStatus>,
        limit: i64,
    ) -> Result<Vec<ReviewCandidate>> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {COLS_CANDIDATE} FROM review_candidates
             WHERE (?1 IS NULL OR subject_id = ?1) AND (?2 IS NULL OR status = ?2)
             ORDER BY created_at DESC, id DESC LIMIT ?3"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![
                subject_id,
                status.map(CandidateStatus::as_str),
                clamp_limit(limit)
            ],
            row_to_candidate,
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub async fn get_candidate(&self, id: &str) -> Result<Option<ReviewCandidate>> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {COLS_CANDIDATE} FROM review_candidates WHERE id = ?1");
        Ok(conn
            .query_row(&sql, params![id], row_to_candidate)
            .optional()?)
    }

    /// Accept or reject a candidate. A CAS on `pending`: `false` means it was
    /// already decided, which the handler turns into a 409 rather than silently
    /// creating a second item from the same fact.
    pub async fn decide_candidate(
        &self,
        id: &str,
        status: CandidateStatus,
        item_id: Option<&str>,
        at: i64,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "UPDATE review_candidates SET status = ?2, item_id = ?3, decided_at = ?4
             WHERE id = ?1 AND status = 'pending'",
            params![id, status.as_str(), item_id, at],
        )?;
        Ok(n > 0)
    }

    pub async fn delete_candidate(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute("DELETE FROM review_candidates WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }
}

// ── Settings ───────────────────────────────────────────────────────────────────

impl TuitionStore {
    /// The spine's knobs, or their defaults when nothing has been written.
    ///
    /// A row that fails to decode also yields the defaults rather than an error:
    /// these are tuning values with documented defaults, and refusing to serve the
    /// app because one of them is unreadable would be the wrong trade.
    pub async fn get_settings(&self) -> Result<TuitionSettings> {
        let conn = self.conn.lock().await;
        let raw: Option<String> = conn
            .query_row(
                "SELECT value FROM settings WHERE scope = ?1",
                params![SETTINGS_SCOPE],
                |r| r.get(0),
            )
            .optional()?;
        Ok(raw
            .and_then(|v| serde_json::from_str(&v).ok())
            .unwrap_or_default())
    }

    pub async fn put_settings(&self, settings: &TuitionSettings) -> Result<()> {
        let value = serde_json::to_string(settings)?;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO settings (scope, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(scope) DO UPDATE SET value = excluded.value,
                                              updated_at = excluded.updated_at",
            params![SETTINGS_SCOPE, value, now_ms()],
        )?;
        Ok(())
    }
}

// ── Event marks ────────────────────────────────────────────────────────────────

impl TuitionStore {
    /// Claim the right to emit one hook event, subject to a cooldown.
    ///
    /// Returns `true` exactly once per cooldown window per `(kind, subject, ref)`.
    /// This is what backs the manifest's promises — `goal.at-risk` "fires at most
    /// once per day per subject", `review.due` "does not fire again for the same
    /// skill until it is reviewed and rescheduled" — and it is a single
    /// conditional UPSERT rather than a read-then-write, so two ticks racing cannot
    /// both claim.
    ///
    /// `kind` is the FULL event id — the `@ryu/tuition#…` constants in
    /// [`crate::state`], not the bare name. The column is opaque, so both forms
    /// "work", which is exactly the problem: claiming with one form and clearing
    /// with the other creates two independent marks and the cooldown silently stops
    /// applying.
    ///
    /// `ref_id` is `""` for subject-scoped events. It is never NULL: SQLite does
    /// not consider two NULLs equal, so a NULL here would defeat the primary key
    /// and the cooldown with it.
    pub async fn claim_event(
        &self,
        kind: &str,
        subject_id: &str,
        ref_id: &str,
        now: i64,
        cooldown_ms: i64,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "INSERT INTO event_marks (kind, subject_id, ref_id, last_emitted_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(kind, subject_id, ref_id) DO UPDATE SET last_emitted_at = ?4
             WHERE ?4 - event_marks.last_emitted_at >= ?5",
            params![kind, subject_id, ref_id, now, cooldown_ms],
        )?;
        Ok(n > 0)
    }

    /// Forget a mark, so the next qualifying condition emits again.
    ///
    /// The counterpart of `review.due`'s "does not fire again for the same skill
    /// until it is reviewed and rescheduled": rescheduling calls this, and without
    /// it a skill that falls due a second time would sit inside its old cooldown
    /// and stay silent.
    pub async fn clear_event_mark(&self, kind: &str, subject_id: &str, ref_id: &str) -> Result<bool> {
        let conn = self.conn.lock().await;
        let n = conn.execute(
            "DELETE FROM event_marks WHERE kind = ?1 AND subject_id = ?2 AND ref_id = ?3",
            params![kind, subject_id, ref_id],
        )?;
        Ok(n > 0)
    }
}

// ── Health ─────────────────────────────────────────────────────────────────────

impl TuitionStore {
    /// Counts for `/health`. Deliberately runs real queries: a liveness probe that
    /// does not touch the database keeps answering 200 while the file underneath it
    /// is unreadable, which is precisely the failure Core's supervisor exists to
    /// catch.
    pub async fn counts(&self) -> Result<StoreCounts> {
        let conn = self.conn.lock().await;
        let one = |sql: &str| -> rusqlite::Result<i64> { conn.query_row(sql, [], |r| r.get(0)) };
        Ok(StoreCounts {
            subjects: one("SELECT COUNT(*) FROM subjects")?,
            skills: one("SELECT COUNT(*) FROM skills")?,
            items: one("SELECT COUNT(*) FROM items")?,
            attempts: one("SELECT COUNT(*) FROM attempts")?,
            pending_candidates: one(
                "SELECT COUNT(*) FROM review_candidates WHERE status = 'pending'",
            )?,
        })
    }
}

// ── Row decoders ───────────────────────────────────────────────────────────────

fn row_to_subject(row: &Row<'_>) -> rusqlite::Result<Subject> {
    Ok(Subject {
        id: row.get(0)?,
        name: row.get(1)?,
        detail: row.get(2)?,
        exam_date: row.get(3)?,
        timezone: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
    })
}

fn row_to_source(row: &Row<'_>) -> rusqlite::Result<Source> {
    let kind: String = row.get(2)?;
    Ok(Source {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        kind: SourceKind::parse(&kind),
        title: row.get(3)?,
        uri: row.get(4)?,
        parsed_text: row.get(5)?,
        parser: row.get(6)?,
        created_at: row.get(7)?,
    })
}

fn row_to_skill(row: &Row<'_>) -> rusqlite::Result<Skill> {
    let status: String = row.get(4)?;
    Ok(Skill {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        name: row.get(2)?,
        detail: row.get(3)?,
        status: SkillStatus::parse(&status),
        source_id: row.get(5)?,
        params: BktParams {
            p_init: row.get(6)?,
            p_transit: row.get(7)?,
            p_slip: row.get(8)?,
            p_guess: row.get(9)?,
        },
        mastery: row.get(10)?,
        ease: row.get(11)?,
        interval_days: row.get(12)?,
        reps: row.get(13)?,
        lapses: row.get(14)?,
        due_at: row.get(15)?,
        last_reviewed_at: row.get(16)?,
        created_at: row.get(17)?,
        updated_at: row.get(18)?,
    })
}

fn row_to_item(row: &Row<'_>) -> rusqlite::Result<Item> {
    let kind: String = row.get(3)?;
    let choices: Option<String> = row.get(5)?;
    let answer: String = row.get(6)?;
    let origin: String = row.get(7)?;
    Ok(Item {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        skill_id: row.get(2)?,
        kind: ItemKind::parse(&kind),
        prompt: row.get(4)?,
        // A choice list that will not decode degrades to none rather than failing
        // the query: an MCQ with no rendered choices is visibly broken, which is
        // better than a subject whose item list refuses to load.
        choices: choices
            .and_then(|c| serde_json::from_str(&c).ok())
            .unwrap_or_default(),
        answer: AnswerKey::decode(&answer),
        origin: ItemOrigin::parse(&origin),
        origin_model: row.get(8)?,
        source_id: row.get(9)?,
        source_ref: row.get(10)?,
        version: row.get(11)?,
        archived: row.get(12)?,
        created_at: row.get(13)?,
        updated_at: row.get(14)?,
    })
}

fn row_to_attempt(row: &Row<'_>) -> rusqlite::Result<Attempt> {
    let graded_by: Option<String> = row.get(9)?;
    Ok(Attempt {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        skill_id: row.get(2)?,
        item_id: row.get(3)?,
        session_id: row.get(4)?,
        item_version: row.get(5)?,
        response: row.get(6)?,
        correct: row.get(7)?,
        score: row.get(8)?,
        graded_by: graded_by.as_deref().map(GradedBy::parse),
        feedback: row.get(10)?,
        latency_ms: row.get(11)?,
        mastery_before: row.get(12)?,
        mastery_after: row.get(13)?,
        informative: row.get(14)?,
        at: row.get(15)?,
    })
}

fn row_to_session(row: &Row<'_>) -> rusqlite::Result<Session> {
    let status: String = row.get(3)?;
    Ok(Session {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        planned_minutes: row.get(2)?,
        status: SessionStatus::parse(&status),
        mastery_gain: row.get(4)?,
        summary: row.get(5)?,
        created_at: row.get(6)?,
        started_at: row.get(7)?,
        finished_at: row.get(8)?,
    })
}

fn row_to_session_item(row: &Row<'_>) -> rusqlite::Result<SessionItem> {
    Ok(SessionItem {
        session_id: row.get(0)?,
        position: row.get(1)?,
        item_id: row.get(2)?,
        skill_id: row.get(3)?,
        est_cost_ms: row.get(4)?,
        est_gain: row.get(5)?,
        attempt_id: row.get(6)?,
    })
}

fn row_to_candidate(row: &Row<'_>) -> rusqlite::Result<ReviewCandidate> {
    let status: String = row.get(7)?;
    Ok(ReviewCandidate {
        id: row.get(0)?,
        subject_id: row.get(1)?,
        conversation_id: row.get(2)?,
        agent_id: row.get(3)?,
        prompt: row.get(4)?,
        answer: row.get(5)?,
        skill_label: row.get(6)?,
        status: CandidateStatus::parse(&status),
        source_key: row.get(8)?,
        source_index: row.get(9)?,
        item_id: row.get(10)?,
        created_at: row.get(11)?,
        decided_at: row.get(12)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn store() -> TuitionStore {
        TuitionStore::open_in_memory().expect("in-memory store")
    }

    /// A subject with one active skill and one item on it — the smallest fixture
    /// anything downstream of `skills` needs.
    async fn seeded(s: &TuitionStore) -> (Subject, Skill, Item) {
        let subject = s
            .create_subject("Pharmacology II", None, Some("2026-09-14"), Some("Europe/Berlin"))
            .await
            .unwrap();
        let skill = s
            .upsert_skill(
                &subject.id,
                "Beta-blocker contraindications",
                None,
                SkillStatus::Active,
                None,
                BktParams::default(),
            )
            .await
            .unwrap();
        let item = s
            .create_item(&NewItem {
                skill_id: skill.id.clone(),
                kind: ItemKind::Mcq,
                prompt: "Which is an absolute contraindication?".into(),
                choices: vec![
                    Choice {
                        id: "a".into(),
                        text: "Severe asthma".into(),
                    },
                    Choice {
                        id: "b".into(),
                        text: "Mild hypertension".into(),
                    },
                ],
                answer: AnswerKey::Mcq {
                    choice_id: "a".into(),
                },
                origin: ItemOrigin::Model,
                origin_model: Some("local/fast".into()),
                source_id: None,
                source_ref: Some("ch. 4, p. 88".into()),
            })
            .await
            .unwrap();
        (subject, skill, item)
    }

    /// The one test that actually EXECUTES the DDL. `cargo check` cannot: a typo in
    /// `V1_DDL` is a string literal and compiles perfectly, then panics on the first
    /// real open. Run this before anything else in this crate.
    #[tokio::test]
    async fn migrations_apply_on_a_fresh_db() {
        let s = store().await;
        assert!(s.list_subjects().await.unwrap().is_empty());
        let counts = s.counts().await.unwrap();
        assert_eq!(counts.subjects, 0);
        assert_eq!(counts.pending_candidates, 0);
        // Settings are served from their defaults with no row present.
        assert_eq!(s.get_settings().await.unwrap(), TuitionSettings::default());
        // And every table named in the cascade exists — a `DELETE FROM` against a
        // missing table is the failure a DDL typo actually produces.
        s.purge_all().await.unwrap();
    }

    /// `migrate()` runs on EVERY open, so an arm that can fail does not fail once —
    /// it bricks the app forever, on exactly the databases the fix was meant to
    /// repair. Two halves, because each catches a different way of getting it wrong:
    ///
    /// 1. Replaying `V1_DDL` on a connection that already has the schema must be a
    ///    no-op. This is what the `IF NOT EXISTS` on every table AND index buys.
    /// 2. Rewinding `user_version` to 0 with real rows present and re-migrating must
    ///    succeed and leave the rows alone. This is the half that would catch a
    ///    future arm adding a constraint over data that violates it.
    #[tokio::test]
    async fn every_migration_arm_is_safe_to_re_run() {
        let conn = Connection::open_in_memory().unwrap();
        TuitionStore::prepare(&conn).unwrap();
        conn.execute_batch(V1_DDL)
            .expect("V1_DDL replays cleanly over itself");

        conn.execute(
            "INSERT INTO subjects (id, name, detail, exam_date, timezone, created_at, updated_at)
             VALUES ('sub_1', 'Anatomy', NULL, NULL, 'UTC', 1, 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO skills (id, subject_id, name, detail, status, source_id, p_init,
                                 p_transit, p_slip, p_guess, mastery, ease, interval_days,
                                 reps, lapses, due_at, last_reviewed_at, created_at, updated_at)
             VALUES ('skl_1', 'sub_1', 'Bones', NULL, 'active', NULL, 0.2, 0.15, 0.1, 0.2,
                     0.2, 2.5, 0, 0, 0, NULL, NULL, 1, 1)",
            [],
        )
        .unwrap();

        conn.pragma_update(None, "user_version", 0).unwrap();
        TuitionStore::migrate(&conn).expect("re-migrating over live rows must not fail");

        let subjects: i64 = conn
            .query_row("SELECT COUNT(*) FROM subjects", [], |r| r.get(0))
            .unwrap();
        let skills: i64 = conn
            .query_row("SELECT COUNT(*) FROM skills", [], |r| r.get(0))
            .unwrap();
        assert_eq!((subjects, skills), (1, 1), "the rows survived the re-run");
        let version: i32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn subjects_round_trip_including_their_last_column() {
        // Deliberately asserts fields at the END of `COLS_SUBJECT`: an off-by-one
        // between the column list and the decoder is invisible to a test that only
        // reads the first two.
        let s = store().await;
        let created = s
            .create_subject("Anatomy", Some("Second year"), Some("2026-12-01"), Some("Asia/Tokyo"))
            .await
            .unwrap();
        let fetched = s.get_subject(&created.id).await.unwrap().unwrap();
        assert_eq!(fetched, created);
        assert_eq!(fetched.exam_date.as_deref(), Some("2026-12-01"));
        assert_eq!(fetched.timezone, "Asia/Tokyo");
        assert_eq!(fetched.updated_at, created.created_at);

        assert!(s
            .update_subject(&created.id, "Anatomy I", None, None, "UTC")
            .await
            .unwrap());
        let updated = s.get_subject(&created.id).await.unwrap().unwrap();
        assert_eq!(updated.name, "Anatomy I");
        // Clearing the exam date is meaningful: no date means no trajectory and no
        // at-risk warning, so it must actually clear.
        assert_eq!(updated.exam_date, None);
        assert!(!s
            .update_subject("sub_missing", "x", None, None, "UTC")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn a_repeated_skill_name_upserts_instead_of_splitting_the_posterior() {
        let s = store().await;
        let (subject, skill, _) = seeded(&s).await;
        // The Study-mode hook reuses its skill labels verbatim, so this is the
        // common path, not an edge case.
        let again = s
            .upsert_skill(
                &subject.id,
                "Beta-blocker contraindications",
                None,
                SkillStatus::Active,
                None,
                BktParams::default(),
            )
            .await
            .unwrap();
        assert_eq!(again.id, skill.id);
        assert_eq!(s.list_skills(&subject.id, None).await.unwrap().len(), 1);
        // Seeded from the prior, not from zero.
        assert_eq!(again.mastery, DEFAULT_P_INIT);
        assert_eq!(again.ease, DEFAULT_EASE);
    }

    #[tokio::test]
    async fn an_upsert_promotes_a_proposed_skill_but_never_demotes_or_forgets() {
        let s = store().await;
        let subject = s.create_subject("Ingest", None, None, None).await.unwrap();
        // Ingest proposes it, with a source and a detail.
        let proposed = s
            .upsert_skill(
                &subject.id,
                "Ion channels",
                Some("from chapter 3"),
                SkillStatus::Proposed,
                Some("src_1"),
                BktParams::default(),
            )
            .await
            .unwrap();
        assert_eq!(proposed.status, SkillStatus::Proposed);

        // Accepting a Study-mode candidate under the same label must ACTIVATE it.
        // Without that, the accepted item hangs off a skill `list_due_skills`
        // filters out and never comes up for review.
        let accepted = s
            .upsert_skill(
                &subject.id,
                "Ion channels",
                None,
                SkillStatus::Active,
                None,
                BktParams::default(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.id, proposed.id);
        assert_eq!(accepted.status, SkillStatus::Active);
        // …and the thinner second offer did not blank what the first recorded.
        assert_eq!(accepted.detail.as_deref(), Some("from chapter 3"));
        assert_eq!(accepted.source_id.as_deref(), Some("src_1"));

        // Learned state is never touched by an upsert — the whole reason
        // re-offering a name is safe.
        s.update_skill_mastery(&proposed.id, 0.77).await.unwrap();
        let reproposed = s
            .upsert_skill(
                &subject.id,
                "Ion channels",
                None,
                SkillStatus::Proposed,
                None,
                BktParams::default(),
            )
            .await
            .unwrap();
        assert_eq!(reproposed.mastery, 0.77, "a re-ingest must not reset it");
        assert_eq!(
            reproposed.status,
            SkillStatus::Active,
            "and must not demote an active skill back to proposed"
        );
    }

    #[tokio::test]
    async fn a_prerequisite_cycle_is_refused_at_write_time_and_names_the_path() {
        let s = store().await;
        let (subject, a, _) = seeded(&s).await;
        let b = s
            .upsert_skill(&subject.id, "B", None, SkillStatus::Active, None, BktParams::default())
            .await
            .unwrap();
        let c = s
            .upsert_skill(&subject.id, "C", None, SkillStatus::Active, None, BktParams::default())
            .await
            .unwrap();
        // A needs B, B needs C.
        s.add_prereq(&a.id, &b.id).await.unwrap();
        s.add_prereq(&b.id, &c.id).await.unwrap();
        // C needing A would close A → B → C → A.
        let err = s.add_prereq(&c.id, &a.id).await.unwrap_err().to_string();
        assert!(err.contains("cycle"), "{err}");
        assert!(err.contains("Beta-blocker contraindications"), "{err}");
        assert!(s.add_prereq(&a.id, &a.id).await.is_err(), "self-edge");
        // The refused edge was not written.
        assert!(s.list_prereqs(&c.id).await.unwrap().is_empty());
        assert_eq!(s.list_prereq_edges(&subject.id).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_prerequisite_must_be_in_the_same_subject() {
        let s = store().await;
        let (_subject, skill, _) = seeded(&s).await;
        let other = s.create_subject("Other", None, None, None).await.unwrap();
        let foreign = s
            .upsert_skill(&other.id, "Foreign", None, SkillStatus::Active, None, BktParams::default())
            .await
            .unwrap();
        // The delete cascade deletes edges by their denormalized `subject_id`, so a
        // cross-subject edge would survive its own skills being removed.
        assert!(s.add_prereq(&skill.id, &foreign.id).await.is_err());
        assert!(s.add_prereq(&skill.id, "skl_ghost").await.is_err());
    }

    #[tokio::test]
    async fn an_item_whose_answer_key_grades_another_kind_is_refused() {
        let s = store().await;
        let (_, skill, _) = seeded(&s).await;
        let wrong = s
            .create_item(&NewItem {
                skill_id: skill.id.clone(),
                kind: ItemKind::Numeric,
                prompt: "How many?".into(),
                choices: vec![],
                answer: AnswerKey::Exact {
                    text: "four".into(),
                    alternatives: vec![],
                },
                origin: ItemOrigin::Human,
                origin_model: None,
                source_id: None,
                source_ref: None,
            })
            .await;
        assert!(wrong.is_err(), "an ungradeable item must not be storable");
    }

    #[tokio::test]
    async fn items_round_trip_their_answer_key_and_derive_their_subject() {
        let s = store().await;
        let (subject, skill, item) = seeded(&s).await;
        // `subject_id` is read off the skill, never supplied — the whole point of
        // the denormalization being safe.
        assert_eq!(item.subject_id, subject.id);
        assert_eq!(item.version, 1);
        assert_eq!(item.source_ref.as_deref(), Some("ch. 4, p. 88"));
        let fetched = s.get_item(&item.id).await.unwrap().unwrap();
        assert_eq!(fetched, item);
        assert_eq!(fetched.choices.len(), 2);
        assert_eq!(
            fetched.answer,
            AnswerKey::Mcq {
                choice_id: "a".into()
            }
        );

        // An edit bumps the version, so a past attempt still names what it answered.
        assert!(s
            .update_item(
                &item.id,
                "Which is absolutely contraindicated?",
                &fetched.choices,
                &AnswerKey::Mcq {
                    choice_id: "b".into()
                },
                None,
            )
            .await
            .unwrap());
        let edited = s.get_item(&item.id).await.unwrap().unwrap();
        assert_eq!(edited.version, 2);
        // …and the kind cannot be changed out from under stored attempts.
        assert!(s
            .update_item(
                &item.id,
                "x",
                &[],
                &AnswerKey::Free {
                    rubric: "anything".into()
                },
                None
            )
            .await
            .is_err());
        assert_eq!(s.list_items(&skill.id, false, 0).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_malformed_answer_survives_reading_but_blocks_rewriting() {
        let s = store().await;
        let (_, _, item) = seeded(&s).await;
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE items SET answer = '{oops' WHERE id = ?1",
                params![item.id],
            )
            .unwrap();
        }
        // The list still loads — one corrupt row must not blank the deck…
        let fetched = s.get_item(&item.id).await.unwrap().unwrap();
        assert!(matches!(fetched.answer, AnswerKey::Malformed { .. }));
        // …but a read-modify-write cannot overwrite the recoverable original with
        // the shell of it.
        assert!(s
            .update_item(&item.id, "p", &[], &fetched.answer, None)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn attempts_take_their_subject_and_version_from_the_item() {
        let s = store().await;
        let (subject, skill, item) = seeded(&s).await;
        let attempt = s
            .record_attempt(&NewAttempt {
                skill_id: skill.id.clone(),
                item_id: item.id.clone(),
                session_id: None,
                response: "a".into(),
                correct: Some(true),
                score: None,
                graded_by: Some(GradedBy::Deterministic),
                feedback: None,
                latency_ms: Some(4_200),
                mastery_before: 0.20,
                mastery_after: 0.61,
                informative: true,
            })
            .await
            .unwrap();
        assert_eq!(attempt.subject_id, subject.id);
        assert_eq!(attempt.item_version, item.version);
        // The last columns of `COLS_ATTEMPT`, which a first-two-fields assertion
        // would never reach.
        assert!(attempt.informative);
        assert!(attempt.at > 0);
        assert_eq!(attempt.graded_by, Some(GradedBy::Deterministic));

        let round_tripped = s.get_attempt(&attempt.id).await.unwrap().unwrap();
        assert_eq!(round_tripped, attempt);
        assert_eq!(
            s.list_attempts_for_skill(&skill.id, 0).await.unwrap().len(),
            1
        );
        assert_eq!(
            s.latencies_for_kind(&subject.id, ItemKind::Mcq, 0)
                .await
                .unwrap(),
            vec![4_200]
        );
        // An attempt filed under the wrong skill would move the wrong posterior.
        assert!(s
            .record_attempt(&NewAttempt {
                skill_id: "skl_other".into(),
                item_id: item.id.clone(),
                session_id: None,
                response: "a".into(),
                correct: Some(true),
                score: None,
                graded_by: None,
                feedback: None,
                latency_ms: None,
                mastery_before: 0.2,
                mastery_after: 0.2,
                informative: true,
            })
            .await
            .is_err());
    }

    #[tokio::test]
    async fn a_rubric_grade_lands_exactly_once() {
        let s = store().await;
        let (_, skill, item) = seeded(&s).await;
        let attempt = s
            .record_attempt(&NewAttempt {
                skill_id: skill.id.clone(),
                item_id: item.id.clone(),
                session_id: None,
                response: "a long free-response answer".into(),
                // Ungraded: the rubric mark has not come back yet.
                correct: None,
                score: None,
                graded_by: None,
                feedback: None,
                latency_ms: None,
                mastery_before: 0.20,
                mastery_after: 0.20,
                informative: true,
            })
            .await
            .unwrap();
        assert_eq!(attempt.correct, None);
        assert!(s
            .grade_attempt(
                &attempt.id,
                true,
                Some(0.8),
                GradedBy::Rubric,
                Some("covers three of four rubric points"),
                0.55,
                true,
            )
            .await
            .unwrap());
        // A retried grading call must not move the posterior a second time.
        assert!(!s
            .grade_attempt(&attempt.id, false, None, GradedBy::Rubric, None, 0.1, true)
            .await
            .unwrap());
        let graded = s.get_attempt(&attempt.id).await.unwrap().unwrap();
        assert_eq!(graded.correct, Some(true));
        assert_eq!(graded.mastery_after, 0.55);
        assert_eq!(graded.graded_by, Some(GradedBy::Rubric));
    }

    #[tokio::test]
    async fn a_session_plan_is_ordered_and_each_slot_is_answered_once() {
        let s = store().await;
        let (subject, skill, item) = seeded(&s).await;
        let second = s
            .create_item(&NewItem {
                skill_id: skill.id.clone(),
                kind: ItemKind::Exact,
                prompt: "Name the receptor.".into(),
                choices: vec![],
                answer: AnswerKey::Exact {
                    text: "beta-1".into(),
                    alternatives: vec!["β1".into()],
                },
                origin: ItemOrigin::Human,
                origin_model: None,
                source_id: None,
                source_ref: None,
            })
            .await
            .unwrap();
        let session = s
            .create_session(
                &subject.id,
                20,
                &[
                    PlannedItem {
                        item_id: item.id.clone(),
                        skill_id: skill.id.clone(),
                        est_cost_ms: 60_000,
                        est_gain: 0.31,
                    },
                    PlannedItem {
                        item_id: second.id.clone(),
                        skill_id: skill.id.clone(),
                        est_cost_ms: 45_000,
                        est_gain: 0.12,
                    },
                ],
            )
            .await
            .unwrap();
        let plan = s.list_session_items(&session.id).await.unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].position, 0);
        assert_eq!(plan[0].item_id, item.id);
        assert_eq!(plan[1].est_cost_ms, 45_000);

        assert!(s.start_session(&session.id, 1_000).await.unwrap());
        // A second Start must not reset the clock the elapsed time is measured from.
        assert!(!s.start_session(&session.id, 9_000).await.unwrap());

        let next = s.next_session_item(&session.id).await.unwrap().unwrap();
        assert_eq!(next.item_id, item.id);
        assert!(s
            .bind_session_attempt(&session.id, &item.id, "att_1")
            .await
            .unwrap());
        // A resubmitted answer must not overwrite the one that already counted.
        assert!(!s
            .bind_session_attempt(&session.id, &item.id, "att_2")
            .await
            .unwrap());
        assert_eq!(
            s.next_session_item(&session.id).await.unwrap().unwrap().item_id,
            second.id
        );

        assert!(s
            .finish_session(&session.id, SessionStatus::Finished, Some(0.18), Some("ok"), 2_000)
            .await
            .unwrap());
        // Finishing twice would double-count the gain the trajectory averages.
        assert!(!s
            .finish_session(&session.id, SessionStatus::Finished, Some(9.9), None, 3_000)
            .await
            .unwrap());
        let recent = s
            .list_recent_finished_sessions(&subject.id, 10)
            .await
            .unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].mastery_gain, Some(0.18));
        assert_eq!(recent[0].finished_at, Some(2_000));
    }

    #[tokio::test]
    async fn a_redelivered_candidate_payload_is_a_no_op() {
        let s = store().await;
        let (subject, _, _) = seeded(&s).await;
        let envelope = CandidateEnvelope {
            v: 1,
            subject_id: subject.id.clone(),
            conversation_id: Some("conv_1".into()),
            agent_id: None,
            created_at: Some("2026-08-10T06:00:00.000Z".into()),
            candidates: vec![
                NewCandidate {
                    prompt: "What does a beta-blocker do?".into(),
                    answer: "Blocks beta-adrenergic receptors.".into(),
                    skill: Some("Beta blockers".into()),
                },
                NewCandidate {
                    prompt: "Name one contraindication.".into(),
                    answer: "Severe asthma.".into(),
                    skill: Some("Beta blockers".into()),
                },
            ],
        };
        let key = "candidate:sub_1:conv_1:abc";
        assert_eq!(s.enqueue_candidates(key, &envelope).await.unwrap(), 2);
        // The drain is keys → get → delete; a crash in between replays the key.
        assert_eq!(s.enqueue_candidates(key, &envelope).await.unwrap(), 0);
        let pending = s
            .list_candidates(Some(&subject.id), Some(CandidateStatus::Pending), 0)
            .await
            .unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].source_key, key);
        assert_eq!(s.counts().await.unwrap().pending_candidates, 2);

        // An unrecognized payload version is refused rather than guessed at.
        let future = CandidateEnvelope { v: 99, ..envelope };
        assert!(s.enqueue_candidates("candidate:x:y:z", &future).await.is_err());
    }

    #[tokio::test]
    async fn candidates_for_an_unknown_subject_are_refused() {
        let s = store().await;
        // `subject_id` here comes from a free-text preference the learner typed; the
        // hook sanitizes its charset and nothing else. A typo must not file
        // candidates somewhere no list names and no delete reaches.
        let orphan = CandidateEnvelope {
            v: 1,
            subject_id: "sub_typo".into(),
            conversation_id: None,
            agent_id: None,
            created_at: None,
            candidates: vec![NewCandidate {
                prompt: "p".into(),
                answer: "a".into(),
                skill: None,
            }],
        };
        assert!(s.enqueue_candidates("candidate:k:3", &orphan).await.is_err());
        assert_eq!(s.counts().await.unwrap().pending_candidates, 0);
    }

    #[tokio::test]
    async fn a_candidate_can_only_be_decided_once() {
        let s = store().await;
        let (subject, _, item) = seeded(&s).await;
        let envelope = CandidateEnvelope {
            v: 1,
            subject_id: subject.id.clone(),
            conversation_id: None,
            agent_id: None,
            created_at: None,
            candidates: vec![NewCandidate {
                prompt: "p".into(),
                answer: "a".into(),
                skill: None,
            }],
        };
        s.enqueue_candidates("candidate:k:1", &envelope).await.unwrap();
        let candidate = s.list_candidates(None, None, 0).await.unwrap().remove(0);
        assert!(s
            .decide_candidate(&candidate.id, CandidateStatus::Accepted, Some(&item.id), 5)
            .await
            .unwrap());
        assert!(!s
            .decide_candidate(&candidate.id, CandidateStatus::Rejected, None, 6)
            .await
            .unwrap());
        let decided = s.get_candidate(&candidate.id).await.unwrap().unwrap();
        assert_eq!(decided.status, CandidateStatus::Accepted);
        assert_eq!(decided.item_id.as_deref(), Some(item.id.as_str()));
        assert_eq!(decided.decided_at, Some(5));
    }

    #[tokio::test]
    async fn due_skills_include_new_work_and_overdue_reviews_but_never_proposed_ones() {
        let s = store().await;
        let (subject, skill, _) = seeded(&s).await;
        // This test used to assert the opposite — that a never-scheduled skill is NOT
        // due — on the reasoning that `review.due` must not announce a hundred new
        // skills after an ingest. That concern is real, but it was enforced in the
        // wrong place: this query also feeds the session PLANNER, so excluding new
        // work here meant nothing was ever planned for a fresh skill, it never earned
        // an attempt, and it never earned a due date. The app was unusable for new
        // material and every test passed.
        //
        // The event-level concern is now handled where it belongs, in `tick.rs`, which
        // filters `review.due` down to skills that have an actual `due_at`.
        let fresh = s.list_due_skills(None, now_ms(), 0).await.unwrap();
        assert!(
            fresh.iter().any(|k| k.id == skill.id),
            "new material must be plannable"
        );

        assert!(s
            .update_skill_schedule(&skill.id, 2.5, 1, 1, 0, 1_000, 500)
            .await
            .unwrap());
        let due = s.list_due_skills(Some(&subject.id), 2_000, 0).await.unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].due_at, Some(1_000));
        assert_eq!(due[0].reps, 1);
        assert_eq!(due[0].last_reviewed_at, Some(500));
        // Not yet due as of an earlier `now` — as a REVIEW. It is still returned as
        // new work would be, so the assertion is on the schedule, not on emptiness.
        let early = s.list_due_skills(None, 999, 0).await.unwrap();
        assert!(
            !early.iter().any(|k| k.id == skill.id && k.due_at == Some(1_000)),
            "a review scheduled for later must not be reported as due now"
        );
        // A proposed skill is invisible to scheduling entirely.
        assert!(s
            .update_skill(&skill.id, &skill.name, None, SkillStatus::Proposed, BktParams::default())
            .await
            .unwrap());
        // A proposed skill is invisible to scheduling entirely, due date or not:
        // it has not been accepted into the syllabus yet.
        assert!(s.list_due_skills(None, 2_000, 0).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_event_claim_holds_for_its_cooldown_and_can_be_cleared() {
        let s = store().await;
        let (subject, skill, _) = seeded(&s).await;
        let day = 24 * 3_600_000;
        // The FULL event id, from the constants the emitter uses. Claiming with the
        // bare name and clearing with the namespaced one would make two independent
        // marks, and the cooldown would quietly stop applying.
        let at_risk = crate::state::EVENT_GOAL_AT_RISK;
        let review_due = crate::state::EVENT_REVIEW_DUE;
        // `ref_id` is "" for the subject-scoped events. If it were NULL the primary
        // key would not collide and the cooldown would do nothing at all.
        assert!(s.claim_event(at_risk, &subject.id, "", 0, day).await.unwrap());
        assert!(!s
            .claim_event(at_risk, &subject.id, "", day - 1, day)
            .await
            .unwrap());
        assert!(s
            .claim_event(at_risk, &subject.id, "", day, day)
            .await
            .unwrap());
        // A different event and a different ref are different marks.
        assert!(s
            .claim_event(review_due, &subject.id, &skill.id, 0, day)
            .await
            .unwrap());
        assert!(!s
            .claim_event(review_due, &subject.id, &skill.id, 1, day)
            .await
            .unwrap());
        // Rescheduling clears it, so the next time it falls due it speaks again.
        assert!(s
            .clear_event_mark(review_due, &subject.id, &skill.id)
            .await
            .unwrap());
        assert!(s
            .claim_event(review_due, &subject.id, &skill.id, 2, day)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn settings_round_trip_and_a_corrupt_row_serves_the_defaults() {
        let s = store().await;
        let mut settings = TuitionSettings::default();
        settings.ready_threshold = 0.75;
        settings.per_skill_item_cap = 3;
        s.put_settings(&settings).await.unwrap();
        assert_eq!(s.get_settings().await.unwrap(), settings);

        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE settings SET value = 'not json' WHERE scope = ?1",
                params![SETTINGS_SCOPE],
            )
            .unwrap();
        }
        // Tuning values with documented defaults: refusing to serve the app because
        // one of them is unreadable would be the wrong trade.
        assert_eq!(s.get_settings().await.unwrap(), TuitionSettings::default());
    }

    #[tokio::test]
    async fn deleting_a_subject_cascades_in_the_documented_order() {
        let s = store().await;
        let (subject, skill, item) = seeded(&s).await;
        let other = s
            .upsert_skill(&subject.id, "Other", None, SkillStatus::Active, None, BktParams::default())
            .await
            .unwrap();
        s.add_prereq(&skill.id, &other.id).await.unwrap();
        let session = s
            .create_session(
                &subject.id,
                15,
                &[PlannedItem {
                    item_id: item.id.clone(),
                    skill_id: skill.id.clone(),
                    est_cost_ms: 60_000,
                    est_gain: 0.2,
                }],
            )
            .await
            .unwrap();
        s.record_attempt(&NewAttempt {
            skill_id: skill.id.clone(),
            item_id: item.id.clone(),
            session_id: Some(session.id.clone()),
            response: "a".into(),
            correct: Some(true),
            score: None,
            graded_by: Some(GradedBy::Deterministic),
            feedback: None,
            latency_ms: Some(1_000),
            mastery_before: 0.2,
            mastery_after: 0.6,
            informative: true,
        })
        .await
        .unwrap();
        s.claim_event("review.due", &subject.id, &skill.id, 0, 1)
            .await
            .unwrap();
        s.enqueue_candidates(
            "candidate:k:2",
            &CandidateEnvelope {
                v: 1,
                subject_id: subject.id.clone(),
                conversation_id: None,
                agent_id: None,
                created_at: None,
                candidates: vec![NewCandidate {
                    prompt: "p".into(),
                    answer: "a".into(),
                    skill: None,
                }],
            },
        )
        .await
        .unwrap();

        assert!(s.delete_subject(&subject.id).await.unwrap());
        assert!(s.get_subject(&subject.id).await.unwrap().is_none());
        assert!(s.get_item(&item.id).await.unwrap().is_none());
        assert!(s.get_session(&session.id).await.unwrap().is_none());
        assert!(s.list_session_items(&session.id).await.unwrap().is_empty());
        assert!(s.list_prereq_edges(&subject.id).await.unwrap().is_empty());
        assert!(s.list_skills(&subject.id, None).await.unwrap().is_empty());
        assert!(s.list_candidates(None, None, 0).await.unwrap().is_empty());
        let counts = s.counts().await.unwrap();
        assert_eq!(
            (counts.subjects, counts.skills, counts.items, counts.attempts),
            (0, 0, 0, 0)
        );
        assert!(!s.delete_subject(&subject.id).await.unwrap());
    }

    #[tokio::test]
    async fn deleting_a_skill_takes_its_edges_from_both_directions() {
        let s = store().await;
        let (subject, a, _) = seeded(&s).await;
        let b = s
            .upsert_skill(&subject.id, "B", None, SkillStatus::Active, None, BktParams::default())
            .await
            .unwrap();
        s.add_prereq(&a.id, &b.id).await.unwrap();
        assert!(s.delete_skill(&b.id).await.unwrap());
        // An edge naming a skill that no longer exists would leave `ready_skills`
        // waiting on a prerequisite that can never be mastered.
        assert!(s.list_prereqs(&a.id).await.unwrap().is_empty());
        assert!(s.list_prereq_edges(&subject.id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn purge_all_empties_every_table_the_data_category_promises() {
        let s = store().await;
        let (subject, skill, item) = seeded(&s).await;
        s.record_attempt(&NewAttempt {
            skill_id: skill.id.clone(),
            item_id: item.id.clone(),
            session_id: None,
            response: "a".into(),
            correct: Some(false),
            score: None,
            graded_by: Some(GradedBy::Deterministic),
            feedback: None,
            latency_ms: None,
            mastery_before: 0.6,
            mastery_after: 0.4,
            informative: true,
        })
        .await
        .unwrap();
        s.put_settings(&TuitionSettings::default()).await.unwrap();

        s.purge_all().await.unwrap();
        let counts = s.counts().await.unwrap();
        assert_eq!(counts.subjects, 0);
        assert_eq!(counts.attempts, 0);
        assert!(s.list_skills(&subject.id, None).await.unwrap().is_empty());
        // Settings are configuration, not study material — the confirm dialog
        // promises the material, so they survive.
        assert_eq!(s.get_settings().await.unwrap(), TuitionSettings::default());
    }

    #[tokio::test]
    async fn list_limits_are_clamped_before_they_reach_sql() {
        assert_eq!(clamp_limit(0), DEFAULT_LIMIT);
        assert_eq!(clamp_limit(-5), DEFAULT_LIMIT);
        assert_eq!(clamp_limit(10), 10);
        assert_eq!(clamp_limit(i64::MAX), MAX_LIMIT);
    }

    #[tokio::test]
    async fn a_brand_new_skill_is_due_immediately() {
        // The bug this asserts against made the whole app dead on arrival, and every
        // other test missed it because they all seeded a due date: a skill created by
        // the learner has `due_at = NULL`, was excluded from `list_due_skills`, so
        // nothing was ever planned for it, so it never earned a due date.
        let store = TuitionStore::open_in_memory().expect("a store");
        let subject = store
            .create_subject("Chemistry", None, None, None)
            .await
            .expect("a subject");
        let fresh = store
            .upsert_skill(
                &subject.id,
                "Mole concept",
                None,
                SkillStatus::Active,
                None,
                BktParams::default(),
            )
            .await
            .expect("a skill");
        assert!(fresh.due_at.is_none(), "a new skill starts with no due date");

        let due = store
            .list_due_skills(Some(&subject.id), 1_786_348_800_000, 50)
            .await
            .expect("a due list");
        assert!(
            due.iter().any(|s| s.id == fresh.id),
            "a never-reviewed skill must be due"
        );
    }

    #[tokio::test]
    async fn an_overdue_review_sorts_before_brand_new_material() {
        // Retention of what has been learned outranks volume of what has not.
        let store = TuitionStore::open_in_memory().expect("a store");
        let now = 1_786_348_800_000;
        let subject = store
            .create_subject("Chemistry", None, None, None)
            .await
            .expect("a subject");
        let new_skill = store
            .upsert_skill(&subject.id, "Aaa new", None, SkillStatus::Active, None, BktParams::default())
            .await
            .expect("a skill");
        let reviewed = store
            .upsert_skill(&subject.id, "Zzz reviewed", None, SkillStatus::Active, None, BktParams::default())
            .await
            .expect("a skill");
        store
            .update_skill_schedule(&reviewed.id, 2.5, 1, 1, 0, now - 1000, now - 1000)
            .await
            .expect("scheduled");

        let due = store
            .list_due_skills(Some(&subject.id), now, 50)
            .await
            .expect("a due list");
        let ids: Vec<&str> = due.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![reviewed.id.as_str(), new_skill.id.as_str()],
            "the overdue review must come first, ahead of the alphabetically earlier new skill"
        );
    }
}
