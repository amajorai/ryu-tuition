//! Exam trajectory — "will I be ready in time", and when to refuse to answer.
//!
//! This is the only forward-looking number in the app, which makes it the easiest
//! thing here to get quietly wrong. A projection built from two data points is
//! numerology, and presenting it next to the measured numbers borrows their
//! credibility. So the refusal case is a first-class result rather than an error:
//! below [`MIN_SESSIONS_FOR_TRAJECTORY`] sessions this returns
//! [`Trajectory::Unknown`] and the UI says it does not know yet.
//!
//! Everything here is arithmetic over recorded sessions. There is no model, and there
//! is no curve fitting — a linear extrapolation of observed mastery gain is already at
//! the limit of what the data supports.

/// Sessions of history required before any projection is made.
pub const MIN_SESSIONS_FOR_TRAJECTORY: usize = 3;

/// How many recent sessions the learning rate is averaged over.
///
/// Ten is a compromise: fewer and one unusually good evening swings the projection,
/// more and a rate measured before a change of approach keeps dragging it.
pub const TRAILING_SESSIONS: usize = 10;

/// Fraction by which projected effort must exceed available effort before the
/// trajectory is called at risk.
///
/// The 15% band exists because the projection is noisy and a warning that fires at
/// 100.4% of capacity is a warning nobody can act on. It also means the alert, when
/// it does fire, is worth interrupting someone for.
pub const AT_RISK_MARGIN: f64 = 0.15;

/// Mastery at or above which a skill needs no further work.
pub const DEFAULT_TARGET_MASTERY: f64 = 0.90;

/// Milliseconds in a day.
const DAY_MS: i64 = 86_400_000;

/// What the projection concluded.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Trajectory {
    /// Not enough history to say anything. NOT an error, and NOT "on track".
    Unknown { sessions: usize, needed: usize },
    /// A projection was made.
    Projected(Projection),
}

/// A made projection, with every input it was made from.
///
/// The inputs ride along so the UI can show its working. A learner told "you are
/// behind" will immediately ask "by how much, and based on what", and recomputing the
/// answer somewhere else in slightly different terms is how two screens end up
/// disagreeing.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Projection {
    /// Total mastery still to be gained across every unmastered skill.
    pub remaining_mastery: f64,
    /// Mean mastery gained per session, over the trailing window.
    pub rate_per_session: f64,
    /// `remaining_mastery / rate_per_session`.
    pub sessions_needed: f64,
    /// Days until the exam, and the sessions those days allow.
    pub days_remaining: i64,
    pub sessions_available: f64,
    pub at_risk: bool,
}

/// Total mastery still to be gained to bring every skill to `target`.
///
/// Takes the masteries rather than the [`crate::models::Skill`] rows on purpose: this
/// module is a pure function and handing it the model type would invite reading a
/// clock, a store or a due date out of it later.
#[must_use]
pub fn remaining_mastery(masteries: &[f64], target: f64) -> f64 {
    masteries
        .iter()
        .map(|mastery| (target - mastery).max(0.0))
        .sum()
}

/// Mean mastery gain per session over the trailing window.
///
/// `gains` is ordered oldest-first; only the last [`TRAILING_SESSIONS`] are used.
/// Negative sessions are kept, not clamped: a session that went badly really is
/// evidence about the rate, and dropping it would bias every projection optimistic.
#[must_use]
pub fn rate_per_session(gains: &[f64]) -> f64 {
    let window = if gains.len() > TRAILING_SESSIONS {
        &gains[gains.len() - TRAILING_SESSIONS..]
    } else {
        gains
    };
    if window.is_empty() {
        return 0.0;
    }
    window.iter().sum::<f64>() / window.len() as f64
}

/// Project whether the syllabus will be covered before the exam.
///
/// `gains` is the per-session mastery gain history, oldest first. `now` and
/// `exam_at` are epoch millis. `sessions_per_day` comes from the learner's settings.
#[must_use]
pub fn project(
    masteries: &[f64],
    gains: &[f64],
    exam_at: i64,
    now: i64,
    sessions_per_day: f64,
    target: f64,
) -> Trajectory {
    if gains.len() < MIN_SESSIONS_FOR_TRAJECTORY {
        return Trajectory::Unknown {
            sessions: gains.len(),
            needed: MIN_SESSIONS_FOR_TRAJECTORY,
        };
    }
    let remaining = remaining_mastery(masteries, target);
    let rate = rate_per_session(gains);
    let days_remaining = ((exam_at - now).max(0)) / DAY_MS;
    let sessions_available = days_remaining as f64 * sessions_per_day.max(0.0);

    // Nothing left to learn is on track by definition, whatever the rate says — and
    // it must be checked FIRST, because a learner who has finished has a rate near
    // zero and would otherwise be told they will never make it.
    if remaining <= 0.0 {
        return Trajectory::Projected(Projection {
            remaining_mastery: 0.0,
            rate_per_session: rate,
            sessions_needed: 0.0,
            days_remaining,
            sessions_available,
            at_risk: false,
        });
    }

    // A non-positive rate with work outstanding cannot be divided into a number of
    // sessions. It is unambiguously at risk — the learner is not currently gaining —
    // and `f64::INFINITY` says exactly that without pretending to a figure.
    let sessions_needed = if rate > 0.0 {
        remaining / rate
    } else {
        f64::INFINITY
    };

    Trajectory::Projected(Projection {
        remaining_mastery: remaining,
        rate_per_session: rate,
        sessions_needed,
        days_remaining,
        sessions_available,
        at_risk: sessions_needed > sessions_available * (1.0 + AT_RISK_MARGIN),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_786_348_800_000;

    fn exam_in(days: i64) -> i64 {
        NOW + days * DAY_MS
    }

    #[test]
    fn below_three_sessions_it_refuses_to_project() {
        // The refusal is the feature. A projection off two points is numerology, and
        // showing it beside the measured numbers borrows their credibility.
        for count in 0..MIN_SESSIONS_FOR_TRAJECTORY {
            let gains = vec![0.1; count];
            let verdict = project(&[0.2], &gains, exam_in(30), NOW, 1.0, DEFAULT_TARGET_MASTERY);
            assert_eq!(
                verdict,
                Trajectory::Unknown {
                    sessions: count,
                    needed: MIN_SESSIONS_FOR_TRAJECTORY
                }
            );
        }
    }

    #[test]
    fn unknown_is_not_on_track() {
        // Asserted explicitly because the tempting shortcut in the UI — "no risk flag
        // means fine" — is exactly wrong here.
        let verdict = project(&[0.0], &[0.1], exam_in(1), NOW, 1.0, DEFAULT_TARGET_MASTERY);
        assert!(!matches!(verdict, Trajectory::Projected(_)));
    }

    #[test]
    fn a_comfortable_pace_is_not_at_risk() {
        // Two skills, 1.6 mastery to gain, 0.2/session → 8 sessions. 30 days at one
        // session a day is 30 available.
        let skills = [0.1, 0.1];
        let gains = [0.2, 0.2, 0.2, 0.2];
        match project(&skills, &gains, exam_in(30), NOW, 1.0, DEFAULT_TARGET_MASTERY) {
            Trajectory::Projected(p) => {
                assert!(!p.at_risk);
                assert!((p.sessions_needed - 8.0).abs() < 1e-9, "{p:?}");
                assert!((p.sessions_available - 30.0).abs() < 1e-9);
            }
            other => panic!("expected a projection, got {other:?}"),
        }
    }

    #[test]
    fn falling_behind_is_flagged() {
        let skills = [0.0, 0.0, 0.0];
        let gains = [0.05, 0.05, 0.05, 0.05];
        // 2.7 to gain at 0.05/session = 54 sessions; 10 days gives 10.
        match project(&skills, &gains, exam_in(10), NOW, 1.0, DEFAULT_TARGET_MASTERY) {
            Trajectory::Projected(p) => assert!(p.at_risk, "{p:?}"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn the_margin_keeps_a_borderline_pace_quiet() {
        // Exactly at capacity must NOT fire: a warning at 100.4% of capacity is one
        // nobody can act on, and it would fire and clear on alternate days.
        let skills = [-0.1]; // remaining = 1.0 after the max(0.0) clamp
        let gains = [0.1, 0.1, 0.1];
        match project(&skills, &gains, exam_in(10), NOW, 1.0, DEFAULT_TARGET_MASTERY) {
            Trajectory::Projected(p) => {
                assert!((p.sessions_needed - 10.0).abs() < 1e-9, "{p:?}");
                assert!(!p.at_risk, "exactly at capacity must stay quiet: {p:?}");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn a_finished_syllabus_is_on_track_even_at_a_zero_rate() {
        // The ordering trap: someone who has finished has a rate near zero, and a
        // naive rate check would tell them they will never make it.
        let skills = [0.95, 1.0];
        let gains = [0.0, 0.0, 0.0];
        match project(&skills, &gains, exam_in(1), NOW, 1.0, DEFAULT_TARGET_MASTERY) {
            Trajectory::Projected(p) => {
                assert!(!p.at_risk);
                assert_eq!(p.remaining_mastery, 0.0);
                assert_eq!(p.sessions_needed, 0.0);
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn no_progress_with_work_left_is_at_risk_and_says_so_without_inventing_a_number() {
        let skills = [0.2];
        let gains = [0.0, 0.0, 0.0, 0.0];
        match project(&skills, &gains, exam_in(30), NOW, 1.0, DEFAULT_TARGET_MASTERY) {
            Trajectory::Projected(p) => {
                assert!(p.at_risk);
                assert!(p.sessions_needed.is_infinite());
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn a_bad_session_is_kept_in_the_rate_not_discarded() {
        // Dropping negative sessions would bias every projection optimistic, which is
        // the one direction this number must not be biased.
        assert!((rate_per_session(&[0.2, -0.1, 0.2]) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn only_the_trailing_window_counts() {
        let mut gains = vec![10.0; 5];
        gains.extend(vec![0.1; TRAILING_SESSIONS]);
        // The ancient 10.0 sessions must fall out of the window entirely.
        assert!((rate_per_session(&gains) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn a_past_exam_date_gives_zero_available_sessions_rather_than_negative() {
        let skills = [0.1];
        let gains = [0.1, 0.1, 0.1];
        match project(&skills, &gains, NOW - 5 * DAY_MS, NOW, 1.0, DEFAULT_TARGET_MASTERY) {
            Trajectory::Projected(p) => {
                assert_eq!(p.days_remaining, 0);
                assert_eq!(p.sessions_available, 0.0);
                assert!(p.at_risk);
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn remaining_mastery_never_counts_an_overshoot_as_negative_work() {
        // A skill above target must not subsidise one below it.
        let skills = [1.0, 0.0];
        assert!((remaining_mastery(&skills, 0.9) - 0.9).abs() < 1e-12);
    }
}
