//! The background tick: drain the hook's queue, and raise the three events.
//!
//! Everything here is best-effort and idempotent. A tick that fails must leave the
//! next one able to do the same work, and a tick that succeeds must not re-raise what
//! it already raised — which is what [`crate::store::TuitionStore::claim_event`] is
//! for: it is a claim, not a check, so two ticks racing cannot both fire.
//!
//! # The three events
//!
//! - `review.due` — once per day, when something is actually due. Fired at most once
//!   per subject per day: a "you have revision waiting" that arrives every five
//!   minutes is one the learner turns off within an hour.
//! - `mastery.dropped` — a skill fell back below the ready threshold it had passed.
//!   The claim is keyed on the skill AND the crossing, so a skill hovering at the
//!   boundary does not fire on every tick.
//! - `goal.at-risk` — the exam projection says the syllabus will not be covered in
//!   time. Requires the three sessions of history the projection itself requires, so
//!   this cannot fire on a subject the learner started yesterday.

use std::time::Duration;

/// Milliseconds in a day.
const DAY_MS: i64 = 86_400_000;

/// How long a `mastery.dropped` claim holds when the skill has NOT recovered.
///
/// Long on purpose. The mark is cleared the instant mastery climbs back over the
/// threshold, so this cooldown only governs the case where it stays below — and there,
/// one notification is the right number until something changes.
const RECOVERY_COOLDOWN_MS: i64 = 30 * DAY_MS;

use crate::{
    models::{now_ms, CandidateEnvelope},
    service,
    state::{AppState, EVENT_GOAL_AT_RISK, EVENT_MASTERY_DROPPED, EVENT_REVIEW_DUE},
    trajectory::Trajectory,
};

/// Run the tick loop until the task is aborted.
pub async fn run(state: AppState) {
    let period = Duration::from_secs(state.config.tick_secs.max(30));
    let mut ticker = tokio::time::interval(period);
    // The first tick fires immediately, which is what makes a freshly spawned sidecar
    // drain a queue the hook filled while it was stopped (it is `lazy`, so that is the
    // normal case, not an edge one).
    loop {
        ticker.tick().await;
        if let Err(err) = once(&state).await {
            // Never propagate: a failing tick must not kill the loop, or one transient
            // error stops every future review notification with nothing in the logs
            // after the first line.
            tracing::warn!(error = %err, "tuition: tick failed");
        }
    }
}

/// One pass. Separated from the loop so it is callable directly in a test.
pub async fn once(state: &AppState) -> anyhow::Result<()> {
    let now = now_ms();
    drain_candidates(state).await;
    raise_review_due(state, now).await;
    raise_mastery_dropped(state, now).await;
    raise_goal_at_risk(state, now).await;
    Ok(())
}

/// Move whatever the Study-mode hook queued in Core's KV into the candidates table.
///
/// The hook cannot reach this process (the sandbox has no HTTP), so this is the only
/// path candidates take. See `host.rs` for why the drain is read-then-delete per key.
async fn drain_candidates(state: &AppState) {
    let Some(host) = state.host.as_ref() else {
        return;
    };
    let drained = match host.drain_candidates(state.config.drain_batch).await {
        Ok(drained) => drained,
        Err(err) => {
            tracing::warn!(error = %err, "tuition: could not drain the candidate queue");
            return;
        }
    };
    for (key, value) in drained {
        let envelope: CandidateEnvelope = match serde_json::from_value(value) {
            Ok(envelope) => envelope,
            Err(err) => {
                // The key is already deleted at this point, so a shape this build
                // cannot read is logged loudly rather than silently dropped.
                tracing::warn!(key, error = %err, "tuition: a queued candidate had an unreadable shape");
                continue;
            }
        };
        match state.store.enqueue_candidates(&key, &envelope).await {
            Ok(count) if count > 0 => {
                tracing::info!(key, count, "tuition: queued review candidates from a chat");
            }
            Ok(_) => {}
            Err(err) => tracing::warn!(key, error = %err, "tuition: could not file candidates"),
        }
    }
}

async fn raise_review_due(state: &AppState, now: i64) {
    let Ok(subjects) = state.store.list_subjects().await else {
        return;
    };
    for subject in subjects {
        let Ok(all_due) = state.store.list_due_skills(Some(&subject.id), now, 100).await else {
            continue;
        };
        // `list_due_skills` deliberately includes never-reviewed skills, because the
        // session planner needs new material. This event does NOT: "your revision is
        // due" announcing a hundred skills the learner has never seen, minutes after
        // an ingest, is the notification that gets the app muted. Only skills with a
        // real `due_at` are an actual review coming back around.
        let due: Vec<_> = all_due.into_iter().filter(|s| s.due_at.is_some()).collect();
        if due.is_empty() {
            continue;
        }
        // Once per subject per calendar day, in the SUBJECT's timezone — the same day
        // boundary the review intervals themselves are counted in. The day is the
        // `ref_id` AND the cooldown is a day: the ref_id alone would re-fire on every
        // tick (a same-day claim is an UPDATE that still reports a row), and the
        // cooldown alone would drift later every day.
        let tz = crate::models::resolve_tz(&subject.timezone);
        let day = crate::models::day_start_ms(tz, now);
        if !claimed(state, "review.due", &subject.id, &day.to_string(), now, DAY_MS).await {
            continue;
        }
        state
            .events
            .emit(
                EVENT_REVIEW_DUE,
                serde_json::json!({
                    "subject_id": subject.id,
                    "subject": subject.name,
                    "due_count": due.len(),
                    "weakest": due.first().map(|s| s.name.clone()),
                    "due_at": now,
                }),
            )
            .await;
    }
}

async fn raise_mastery_dropped(state: &AppState, now: i64) {
    let Ok(settings) = state.store.get_settings().await else {
        return;
    };
    let Ok(subjects) = state.store.list_subjects().await else {
        return;
    };
    for subject in subjects {
        let Ok(skills) = state.store.list_skills(&subject.id, None).await else {
            continue;
        };
        for skill in skills {
            // Only a skill that HAD passed the threshold can drop below it. Without
            // the `reps` check every never-practised skill would be "dropped" on the
            // first tick after it was created.
            if skill.reps == 0 || skill.mastery >= settings.ready_threshold {
                // Clear the mark so a later genuine drop fires again.
                let _ = state
                    .store
                    .clear_event_mark("mastery.dropped", &subject.id, &skill.id)
                    .await;
                continue;
            }
            // Effectively "once until it recovers": the mark is cleared above the
            // moment mastery climbs back over the threshold, so the long cooldown is
            // what stops a skill hovering AT the boundary from firing every tick.
            if !claimed(
                state,
                "mastery.dropped",
                &subject.id,
                &skill.id,
                now,
                RECOVERY_COOLDOWN_MS,
            )
            .await
            {
                continue;
            }
            state
                .events
                .emit(
                    EVENT_MASTERY_DROPPED,
                    serde_json::json!({
                        "subject_id": subject.id,
                        "skill_id": skill.id,
                        "skill": skill.name,
                        "mastery": skill.mastery,
                        "threshold": settings.ready_threshold,
                        "lapses": skill.lapses,
                    }),
                )
                .await;
        }
    }
}

async fn raise_goal_at_risk(state: &AppState, now: i64) {
    let Ok(subjects) = state.store.list_subjects().await else {
        return;
    };
    for subject in subjects {
        if subject.exam_date.is_none() {
            continue;
        }
        let Ok(report) = service::mastery_report(state, &subject.id, now).await else {
            continue;
        };
        let Some(Trajectory::Projected(projection)) = report.trajectory else {
            // `Unknown` is NOT "on track" — it is "we do not know yet" — so it must
            // not clear the mark either. Doing nothing is the whole handling.
            continue;
        };
        if !projection.at_risk {
            let _ = state
                .store
                .clear_event_mark("goal.at-risk", &subject.id, "")
                .await;
            continue;
        }
        // Re-claimable once a day: a learner who is behind stays behind, and a daily
        // nudge is a reminder while an hourly one is noise.
        if !claimed(state, "goal.at-risk", &subject.id, "", now, DAY_MS).await {
            continue;
        }
        state
            .events
            .emit(
                EVENT_GOAL_AT_RISK,
                serde_json::json!({
                    "subject_id": subject.id,
                    "subject": subject.name,
                    "exam_date": subject.exam_date,
                    "days_remaining": projection.days_remaining,
                    "sessions_needed": finite(projection.sessions_needed),
                    "sessions_available": projection.sessions_available,
                    "remaining_mastery": projection.remaining_mastery,
                }),
            )
            .await;
    }
}

/// `f64::INFINITY` is not representable in JSON and `serde_json` turns it into
/// `null`, which reads as "no data" rather than "no progress at all". Sending it as
/// `null` explicitly at least makes that deliberate, and the consumer has
/// `remaining_mastery` to tell the two apart.
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// Claim an event, returning whether THIS caller won it.
async fn claimed(
    state: &AppState,
    kind: &str,
    subject_id: &str,
    ref_id: &str,
    now: i64,
    cooldown_ms: i64,
) -> bool {
    match state
        .store
        .claim_event(kind, subject_id, ref_id, now, cooldown_ms)
        .await
    {
        Ok(won) => won,
        Err(err) => {
            tracing::warn!(kind, subject_id, error = %err, "tuition: could not claim an event");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_infinite_projection_is_reported_as_absent_not_as_a_number() {
        assert_eq!(finite(f64::INFINITY), None);
        assert_eq!(finite(f64::NAN), None);
        assert_eq!(finite(12.0), Some(12.0));
    }

    #[tokio::test]
    async fn a_tick_against_an_empty_store_does_nothing_and_does_not_fail() {
        // The first tick of a freshly installed app. It must be a no-op, not a
        // notification about a subject that does not exist.
        let store = crate::store::TuitionStore::open_in_memory().expect("a store");
        let state = AppState::new(store, crate::state::Config::from_env(8007));
        once(&state).await.expect("an empty tick must succeed");
    }
}
