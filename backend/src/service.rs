//! The operations, in one place, so the HTTP surface and the MCP server cannot drift.
//!
//! Both front ends do the same four things — plan a session, serve what is due, grade
//! an answer, report mastery — and both of them are thin. Putting the pipeline here
//! rather than in `api.rs` is what stops an agent that drills you through
//! `tuition__grade` from updating the mastery model differently than the companion
//! does, which would make the two disagree about what you know.
//!
//! # The grading pipeline, in order
//!
//! 1. [`crate::grade`] decides the answer — with **no model** for every objective kind.
//! 2. Only an *informative* outcome moves the Bayesian posterior. A row we could not
//!    read is recorded and skipped: marking a learner wrong for our own corrupt data
//!    would move the model on evidence that says nothing about what they know.
//! 3. The attempt is written with `mastery_before` and `mastery_after` ON THE ROW, so
//!    the history stays truthful even after the parameters are re-tuned.
//!
//! Review scheduling deliberately does NOT happen here. SM-2 runs over a whole
//! session's outcomes at [`finish_session`], not per answer — a skill is scheduled
//! once from how the session went, and rescheduling it after every item would let one
//! lucky answer erase a lapse.

use anyhow::{anyhow, Result};

use crate::{
    bkt, grade,
    host::{no_host_message, Host},
    models::{
        resolve_tz, Attempt, GradedBy, Item, ItemKind, NewAttempt, PlannedItem, SessionStatus,
        Skill, TuitionSettings,
    },
    planner::{self, Candidate, CostModel, PlanOptions},
    srs,
    state::AppState,
    trajectory::{self, Trajectory},
};

/// How much source material is handed to the generation model.
///
/// A cap rather than the whole library: the context is paid for on every generate,
/// and a syllabus that grew past this would start silently truncating in the middle
/// of a sentence with no indication of where.
pub const MAX_GENERATION_CONTEXT_CHARS: usize = 12_000;

/// What grading one answer did.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnswerResult {
    pub attempt: Attempt,
    pub correct: Option<bool>,
    pub score: Option<f64>,
    pub graded_by: Option<GradedBy>,
    /// Per-blank results for a cloze item; empty otherwise.
    pub blanks: Vec<bool>,
    /// The rubric a free-response answer was marked against. Shown WITH the mark —
    /// a rubric grade the learner cannot see the rubric for is just an opinion.
    pub rubric: Option<String>,
    pub feedback: Option<String>,
    pub mastery_before: f64,
    pub mastery_after: f64,
    /// False when the outcome could not move the posterior, with the reason.
    pub informative: bool,
    pub note: Option<String>,
}

/// Grade one answer and fold it into the mastery model.
///
/// `host` is optional: without it, a free-response answer is recorded UNGRADED rather
/// than refused, so the attempt is not lost and can be marked later. Objective kinds
/// never need it.
pub async fn answer(
    state: &AppState,
    item_id: &str,
    response: &str,
    session_id: Option<&str>,
    host: Option<&Host>,
) -> Result<AnswerResult> {
    let item: Item = state
        .store
        .get_item(item_id)
        .await?
        .ok_or_else(|| anyhow!("no such item"))?;
    let skill: Skill = state
        .store
        .get_skill(&item.skill_id)
        .await?
        .ok_or_else(|| anyhow!("this item's skill is missing"))?;

    let outcome = grade::grade(item.kind, &item.answer, response);
    let mastery_before = skill.mastery;

    // Record the attempt first, ungraded, so a crash between grading and writing
    // cannot lose the fact that the learner answered.
    let attempt = state
        .store
        .record_attempt(&NewAttempt {
            skill_id: item.skill_id.clone(),
            item_id: item.id.clone(),
            session_id: session_id.map(str::to_owned),
            response: response.to_owned(),
            correct: None,
            score: None,
            graded_by: None,
            feedback: None,
            latency_ms: None,
            mastery_before,
            // Equal to `mastery_before` until something grades it: an ungraded
            // attempt has not moved the posterior, and writing the eventual value
            // here would make the row claim it did.
            mastery_after: mastery_before,
            informative: false,
        })
        .await?;

    let mut result = AnswerResult {
        attempt: attempt.clone(),
        correct: None,
        score: None,
        graded_by: None,
        blanks: Vec::new(),
        rubric: None,
        feedback: None,
        mastery_before,
        mastery_after: mastery_before,
        informative: false,
        note: None,
    };

    match outcome {
        grade::Outcome::Decided {
            correct,
            score,
            graded_by,
            blanks,
        } => {
            let update = bkt::update_for_item(
                mastery_before,
                skill.params,
                item.kind,
                item.choices.len(),
                correct,
            );
            finalize(
                state,
                &attempt,
                &skill,
                correct,
                Some(score),
                graded_by,
                None,
                update.after,
                update.informative,
            )
            .await?;
            result.correct = Some(correct);
            result.score = Some(score);
            result.graded_by = Some(graded_by);
            result.blanks = blanks;
            result.mastery_after = update.after;
            result.informative = update.informative;
            if !update.informative {
                result.note =
                    Some("this answer was recorded but did not move your mastery estimate".into());
            }
        }
        grade::Outcome::NeedsRubric { rubric } => {
            result.rubric = Some(rubric.clone());
            let Some(host) = host else {
                result.note = Some(no_host_message().to_owned());
                return Ok(result);
            };
            let settings = state.store.get_settings().await?;
            let marked = mark_against_rubric(host, &item, &rubric, response, &settings).await?;
            let update = bkt::update_for_item(
                mastery_before,
                skill.params,
                item.kind,
                item.choices.len(),
                marked.correct,
            );
            finalize(
                state,
                &attempt,
                &skill,
                marked.correct,
                Some(marked.score),
                GradedBy::Rubric,
                marked.feedback.as_deref(),
                update.after,
                update.informative,
            )
            .await?;
            result.correct = Some(marked.correct);
            result.score = Some(marked.score);
            result.graded_by = Some(GradedBy::Rubric);
            result.feedback = marked.feedback;
            result.mastery_after = update.after;
            result.informative = update.informative;
        }
        grade::Outcome::Ungradeable { reason } => {
            // Deliberately NOT recorded as incorrect, and the posterior does not move.
            result.note = Some(reason);
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
async fn finalize(
    state: &AppState,
    attempt: &Attempt,
    skill: &Skill,
    correct: bool,
    score: Option<f64>,
    graded_by: GradedBy,
    feedback: Option<&str>,
    mastery_after: f64,
    informative: bool,
) -> Result<()> {
    state
        .store
        .grade_attempt(
            &attempt.id,
            correct,
            score,
            graded_by,
            feedback,
            mastery_after,
            informative,
        )
        .await?;
    if informative {
        state
            .store
            .update_skill_mastery(&skill.id, mastery_after)
            .await?;
    }
    Ok(())
}

struct RubricMark {
    correct: bool,
    score: f64,
    feedback: Option<String>,
}

/// Mark a free-response answer against its rubric.
///
/// The prompt asks for a score and one sentence of feedback, and the score is clamped
/// rather than trusted: a model that returns 7 on a 0..1 scale must not produce a
/// mastery posterior above 1.
async fn mark_against_rubric(
    host: &Host,
    item: &Item,
    rubric: &str,
    response: &str,
    settings: &TuitionSettings,
) -> Result<RubricMark> {
    let system = "You mark one short answer against a written rubric. Reply with JSON \
        only: {\"score\": <0..1>, \"feedback\": \"<one sentence>\"}. Score strictly \
        against the rubric and nothing else — not style, not length, not whether you \
        would have said it differently.";
    let prompt = format!(
        "Question:\n{}\n\nRubric:\n{rubric}\n\nAnswer:\n{response}",
        item.prompt
    );
    let raw = host
        .complete(system, &prompt, Some("tuition-grading-model"))
        .await?;
    let parsed: serde_json::Value =
        extract_json(&raw).ok_or_else(|| anyhow!("the grading model did not return JSON"))?;
    let score = parsed
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let feedback = parsed
        .get("feedback")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok(RubricMark {
        correct: score >= settings.target_mastery.min(0.6),
        score,
        feedback,
    })
}

/// Pull the first balanced JSON object out of a model reply.
///
/// Models wrap JSON in prose and fences no matter how firmly the system prompt says
/// not to, so this scans for a balanced object rather than trusting the whole reply to
/// parse. String-aware, so a `}` inside a quoted feedback sentence does not end it.
#[must_use]
pub fn extract_json(text: &str) -> Option<serde_json::Value> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(text.get(start..=index)?).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Record practice that happened OUTSIDE a stored item — a past paper, a lesson, a
/// conversation with an agent.
///
/// It moves the posterior exactly the way a graded attempt does, and it is written as
/// a real attempt row so the history shows where the movement came from. The item id
/// is empty on purpose: inventing a synthetic item would put a question in the deck
/// that the learner was never asked and that `quiz` would later serve.
///
/// The guess rate used is the one for a free-response item, which is the lowest of
/// them — practice reported by an agent should not get the benefit of a
/// four-choice guess floor it never faced.
pub async fn log_practice(
    state: &AppState,
    skill_id: &str,
    correct: bool,
    note: Option<&str>,
) -> Result<AnswerResult> {
    let skill = state
        .store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow!("no such skill"))?;
    let mastery_before = skill.mastery;
    let update = bkt::update_for_item(mastery_before, skill.params, ItemKind::Free, 0, correct);

    let attempt = state
        .store
        .record_attempt(&NewAttempt {
            skill_id: skill.id.clone(),
            item_id: String::new(),
            session_id: None,
            response: note.unwrap_or("practised outside the app").to_owned(),
            correct: Some(correct),
            score: Some(if correct { 1.0 } else { 0.0 }),
            graded_by: Some(GradedBy::Deterministic),
            feedback: note.map(str::to_owned),
            latency_ms: None,
            mastery_before,
            mastery_after: update.after,
            informative: update.informative,
        })
        .await?;
    if update.informative {
        state
            .store
            .update_skill_mastery(&skill.id, update.after)
            .await?;
    }

    Ok(AnswerResult {
        attempt,
        correct: Some(correct),
        score: Some(if correct { 1.0 } else { 0.0 }),
        graded_by: Some(GradedBy::Deterministic),
        blanks: Vec::new(),
        rubric: None,
        feedback: note.map(str::to_owned),
        mastery_before,
        mastery_after: update.after,
        informative: update.informative,
        note: None,
    })
}

/// Plan a study session against a minutes budget.
pub async fn plan_session(
    state: &AppState,
    subject_id: &str,
    budget_minutes: u32,
    now: i64,
) -> Result<Vec<PlannedItem>> {
    let settings = state.store.get_settings().await?;
    let due = state
        .store
        .list_due_skills(Some(subject_id), now, 200)
        .await?;
    if due.is_empty() {
        return Ok(Vec::new());
    }

    // One cost model for the whole plan, seeded from the settings default and
    // refined per kind by the observed median latency. Building it once matters:
    // re-deriving it per candidate would re-read every latency row per item.
    let mut costs = CostModel::seeded(settings.default_item_seconds);
    for kind in [
        ItemKind::Mcq,
        ItemKind::Cloze,
        ItemKind::Numeric,
        ItemKind::Exact,
        ItemKind::Free,
    ] {
        let latencies = state.store.latencies_for_kind(subject_id, kind, 50).await?;
        costs = costs.observe(kind, &latencies);
    }

    let mut items: Vec<(Item, Skill)> = Vec::new();
    for skill in due {
        for item in state.store.list_items(&skill.id, false, 50).await? {
            items.push((item, skill.clone()));
        }
    }
    let candidates: Vec<Candidate<'_>> = items
        .iter()
        .map(|(item, skill)| Candidate {
            item_id: &item.id,
            skill_id: &skill.id,
            kind: item.kind,
            choice_count: item.choices.len(),
            mastery: skill.mastery,
            params: skill.params,
        })
        .collect();

    Ok(planner::plan(
        &candidates,
        &costs,
        PlanOptions {
            budget_minutes,
            per_skill_item_cap: settings.per_skill_item_cap,
        },
    ))
}

/// Close a session: run SM-2 once per skill, from how that skill actually went.
///
/// This is the only place review scheduling happens. Doing it per answer would let one
/// lucky final item erase a lapse recorded three questions earlier.
pub async fn finish_session(state: &AppState, session_id: &str, now: i64) -> Result<usize> {
    let session = state
        .store
        .get_session(session_id)
        .await?
        .ok_or_else(|| anyhow!("no such session"))?;
    // The timezone is the SUBJECT's, not a global setting: day boundaries are what
    // SM-2 intervals are counted in, and a learner studying a course in another zone
    // should get that course's days.
    let tz = subject_tz(state, &session.subject_id).await?;
    let attempts = state.store.list_attempts_for_session(session_id).await?;
    let mut by_skill: std::collections::BTreeMap<String, Vec<srs::Outcome>> =
        std::collections::BTreeMap::new();
    for attempt in &attempts {
        by_skill
            .entry(attempt.skill_id.clone())
            .or_default()
            .push(srs::Outcome::of(attempt));
    }

    let mut rescheduled = 0usize;
    for (skill_id, outcomes) in by_skill {
        let Some(quality) = srs::quality(&outcomes) else {
            // Every attempt for this skill is still awaiting a rubric mark. Leaving
            // the schedule alone is right: rescheduling on no evidence would push the
            // skill out as though it had been reviewed successfully.
            continue;
        };
        let Some(skill) = state.store.get_skill(&skill_id).await? else {
            continue;
        };
        let review = srs::review(
            srs::Schedule {
                ease: skill.ease,
                interval_days: skill.interval_days,
                reps: skill.reps,
                lapses: skill.lapses,
            },
            quality,
            tz,
            now,
        );
        state
            .store
            .update_skill_schedule(
                &skill_id,
                review.schedule.ease,
                review.schedule.interval_days,
                review.schedule.reps,
                review.schedule.lapses,
                review.due_at,
                now,
            )
            .await?;
        rescheduled += 1;
    }
    let gain = mastery_gain(&attempts);
    state
        .store
        .finish_session(session_id, SessionStatus::Finished, gain, None, now)
        .await?;
    Ok(rescheduled)
}

/// Generate practice items for a skill from the subject's own source material.
///
/// One of the app's exactly two model edges. The items are STORED and versioned, so a
/// generated question is reviewable, editable and reusable rather than conjured fresh
/// every session — which also means a bad question can be fixed once instead of
/// recurring.
///
/// Only objective kinds are requested. A generated free-response item would need a
/// generated rubric, and a rubric nobody wrote is exactly the "a model thought so"
/// grading this app exists to avoid.
pub async fn generate_items(
    state: &AppState,
    host: &Host,
    skill_id: &str,
    count: u32,
) -> Result<Vec<crate::models::Item>> {
    let skill = state
        .store
        .get_skill(skill_id)
        .await?
        .ok_or_else(|| anyhow!("no such skill"))?;

    // Ground the generation in the learner's OWN material where there is any. Without
    // it the model writes from general knowledge, which produces questions about a
    // syllabus the learner is not sitting.
    let sources = state.store.list_sources(&skill.subject_id).await?;
    let material: String = sources
        .iter()
        .map(|s| s.parsed_text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
        .chars()
        .take(MAX_GENERATION_CONTEXT_CHARS)
        .collect();

    let system = "You write practice questions from source material. Reply with JSON \
        only: {\"items\": [{\"kind\": \"mcq\"|\"cloze\"|\"numeric\"|\"exact\", \"prompt\": \"...\", \
        \"choices\": [{\"id\": \"a\", \"text\": \"...\"}], \"answer\": {...}}]}. The answer object \
        matches the kind: mcq {\"kind\":\"mcq\",\"choice_id\":\"a\"}; cloze \
        {\"kind\":\"cloze\",\"blanks\":[[\"one\",\"alt\"]]}; numeric \
        {\"kind\":\"numeric\",\"expected\":\"3.14\",\"tolerance\":{\"kind\":\"absolute\",\"value\":\"0.01\"}}; \
        exact {\"kind\":\"exact\",\"text\":\"...\",\"alternatives\":[]}. Ask about the material, \
        not around it. Never write a free-response question.";
    let prompt = format!(
        "Skill: {}\n{}\n\nWrite {count} questions.\n\nSource material:\n{material}",
        skill.name,
        skill.detail.as_deref().unwrap_or("")
    );

    let raw = host
        .complete(system, &prompt, Some("tuition-generation-model"))
        .await?;
    let parsed =
        extract_json(&raw).ok_or_else(|| anyhow!("the generation model did not return JSON"))?;
    let proposed = parsed
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut created = Vec::new();
    for entry in proposed.into_iter().take(count as usize) {
        let kind = ItemKind::parse(entry.get("kind").and_then(|v| v.as_str()).unwrap_or(""));
        if kind == ItemKind::Free {
            // Refused rather than downgraded: see the doc comment.
            continue;
        }
        let Some(prompt_text) = entry.get("prompt").and_then(|v| v.as_str()) else {
            continue;
        };
        let choices: Vec<crate::models::Choice> = entry
            .get("choices")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let Some(answer): Option<crate::models::AnswerKey> = entry
            .get("answer")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
        else {
            continue;
        };
        // A generated item whose answer cannot be graded is worse than no item: it
        // would be served, answered, and then reported ungradeable.
        if matches!(answer, crate::models::AnswerKey::Malformed { .. }) {
            continue;
        }
        let item = state
            .store
            .create_item(&crate::models::NewItem {
                skill_id: skill.id.clone(),
                kind,
                prompt: prompt_text.to_owned(),
                choices,
                answer,
                origin: crate::models::ItemOrigin::Model,
                origin_model: Some("tuition-generation-model".into()),
                source_id: sources.first().map(|s| s.id.clone()),
                source_ref: None,
            })
            .await?;
        created.push(item);
    }
    Ok(created)
}

/// Promote a Study-mode review candidate into a real item.
///
/// The candidate carries a prompt and an answer the hook extracted from a chat. It
/// becomes an `exact`-match item against that answer, with the original text kept as
/// an accepted alternative — the learner is being asked to recall something they were
/// told, not to reproduce it word for word.
pub async fn accept_candidate(
    state: &AppState,
    candidate_id: &str,
    now: i64,
) -> Result<crate::models::Item> {
    let candidate = state
        .store
        .get_candidate(candidate_id)
        .await?
        .ok_or_else(|| anyhow!("no such candidate"))?;

    // The hook proposes a skill by LABEL, not by id — it has no way to read the skill
    // table. Matching by name and creating on a miss is what turns that label into a
    // real skill without the learner having to pre-create one.
    let label = candidate
        .skill_label
        .clone()
        .unwrap_or_else(|| "From your chats".to_owned());
    let skill = state
        .store
        .upsert_skill(
            &candidate.subject_id,
            &label,
            None,
            crate::models::SkillStatus::Active,
            None,
            crate::models::BktParams::default(),
        )
        .await?;

    let item = state
        .store
        .create_item(&crate::models::NewItem {
            skill_id: skill.id,
            kind: ItemKind::Exact,
            prompt: candidate.prompt.clone(),
            choices: Vec::new(),
            answer: crate::models::AnswerKey::Exact {
                text: candidate.answer.clone(),
                alternatives: Vec::new(),
            },
            origin: crate::models::ItemOrigin::Candidate,
            origin_model: None,
            source_id: None,
            source_ref: None,
        })
        .await?;

    state
        .store
        .decide_candidate(
            candidate_id,
            crate::models::CandidateStatus::Accepted,
            Some(&item.id),
            now,
        )
        .await?;
    Ok(item)
}

/// The mastery report for a subject, including the exam projection.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MasteryReport {
    pub skills: Vec<SkillMastery>,
    pub trajectory: Option<Trajectory>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SkillMastery {
    pub skill_id: String,
    pub name: String,
    pub mastery: f64,
    pub due_at: Option<i64>,
    /// How many more correct answers reach the target. A PROJECTION, labelled as one
    /// wherever it is shown.
    pub correct_answers_to_target: Option<u32>,
}

/// Report mastery per skill, and — when there is an exam date and enough history — the
/// trajectory toward it.
pub async fn mastery_report(state: &AppState, subject_id: &str, now: i64) -> Result<MasteryReport> {
    let settings = state.store.get_settings().await?;
    let skills = state.store.list_skills(subject_id, None).await?;
    let target = settings.target_mastery;

    let entries: Vec<SkillMastery> = skills
        .iter()
        .map(|skill| {
            let guess = bkt::guess_rate(skill.params, ItemKind::Exact, 0);
            let projection =
                bkt::correct_answers_to_target(skill.mastery, skill.params, guess, target);
            SkillMastery {
                skill_id: skill.id.clone(),
                name: skill.name.clone(),
                mastery: skill.mastery,
                due_at: skill.due_at,
                correct_answers_to_target: projection.steps,
            }
        })
        .collect();

    let subject = state.store.get_subject(subject_id).await?;
    let exam_at = subject.as_ref().and_then(|s| exam_at_ms(s));
    let trajectory = match exam_at {
        Some(exam_at) => {
            let sessions = state
                .store
                .list_recent_finished_sessions(subject_id, i64::from(settings.trajectory_window))
                .await?;
            // A session with no recorded gain is one that was abandoned before
            // anything was graded. Dropping it is right — counting it as a zero would
            // drag the learning rate down and make the projection pessimistic.
            let gains: Vec<f64> = sessions.iter().filter_map(|s| s.mastery_gain).collect();
            let masteries: Vec<f64> = skills.iter().map(|s| s.mastery).collect();
            Some(trajectory::project(
                &masteries,
                &gains,
                exam_at,
                now,
                settings.sessions_per_day,
                target,
            ))
        }
        None => None,
    };

    Ok(MasteryReport {
        skills: entries,
        trajectory,
    })
}

/// The subject's IANA zone, falling back to the app default when it is unset or was
/// hand-edited to something unresolvable.
async fn subject_tz(state: &AppState, subject_id: &str) -> Result<chrono_tz::Tz> {
    let subject = state.store.get_subject(subject_id).await?;
    Ok(resolve_tz(
        subject.as_ref().map_or("", |s| s.timezone.as_str()),
    ))
}

/// Total mastery gained across a session's graded attempts.
///
/// Only informative attempts count. A session made entirely of ungradeable rows has
/// no gain — reported as `None`, not `Some(0.0)`, so the trajectory can drop it
/// instead of averaging in a zero the learner did not earn.
fn mastery_gain(attempts: &[Attempt]) -> Option<f64> {
    let mut total = 0.0;
    let mut counted = 0usize;
    for attempt in attempts {
        if attempt.informative {
            total += attempt.mastery_after - attempt.mastery_before;
            counted += 1;
        }
    }
    (counted > 0).then_some(total)
}

/// The exam date as epoch millis: the START of that calendar day in the subject's own
/// timezone.
///
/// `exam_date` is stored as the `YYYY-MM-DD` string the learner typed and never as an
/// instant, precisely so it cannot drift a day when a timezone changes. This is the
/// one place it becomes a moment, and it does so against the subject's zone.
fn exam_at_ms(subject: &crate::models::Subject) -> Option<i64> {
    let raw = subject.exam_date.as_deref()?.trim();
    let date = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").ok()?;
    let tz = resolve_tz(&subject.timezone);
    let noon = date.and_hms_opt(12, 0, 0)?;
    // Via noon, then back to the day start: midnight does not exist on a DST spring
    // forward in some zones, and `day_start_ms` already single-sources that handling.
    let anchor = noon.and_utc().timestamp_millis();
    Some(crate::models::day_start_ms(tz, anchor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_recovered_from_a_fenced_reply() {
        // Models wrap JSON in prose and fences however firmly the prompt says not to.
        let raw =
            "Sure!\n```json\n{\"score\": 0.75, \"feedback\": \"Close.\"}\n```\nHope that helps.";
        let parsed = extract_json(raw).expect("must recover the object");
        assert_eq!(parsed["score"], 0.75);
    }

    #[test]
    fn a_brace_inside_a_string_does_not_end_the_object() {
        let raw = r#"{"feedback": "use a } brace", "score": 1}"#;
        let parsed = extract_json(raw).expect("must parse");
        assert_eq!(parsed["score"], 1);
        assert_eq!(parsed["feedback"], "use a } brace");
    }

    #[test]
    fn an_unterminated_object_yields_nothing_rather_than_panicking() {
        assert!(extract_json("{\"score\": 1").is_none());
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("").is_none());
    }
}
