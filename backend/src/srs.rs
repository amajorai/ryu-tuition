//! Spaced repetition — SM-2 over **skills**, not cards.
//!
//! The unit of scheduling is the skill because the unit of knowledge is the skill:
//! a learner who can answer three questions about beta blockers does not owe a
//! review on each of them separately, and a per-card schedule would ask them the
//! same thing four times in a week while the neighbouring skill goes untouched for
//! a month. Items are drawn *from* the skill that is due (see
//! [`crate::planner`]); the schedule state lives on the skill row.
//!
//! # Two things that are easy to get wrong, and are not
//!
//! **The interval is whole days from a day boundary, in the learner's zone.** Not
//! `now + n × 86_400_000`. A session that runs at 23:58 and schedules "one day"
//! must land on tomorrow's boundary, not on tomorrow at 23:58 — otherwise every
//! late-night sitting walks the review an hour further into the evening until it
//! wraps into the following day, and the learner's "daily" review quietly becomes
//! every other day. [`crate::models::day_start_plus_days`] does the calendar
//! arithmetic (including the DST-gap zones where local midnight does not exist);
//! this module only decides how many days.
//!
//! **The quality grade comes from outcomes, not from a feeling.** Classic SM-2 asks
//! the learner to self-rate 0–5, which is exactly the kind of unauditable number
//! this app refuses to build on. Here `q = round(5 × weighted_correct_ratio)` over
//! the session's graded attempts, so the schedule is derived from what was answered.

use chrono_tz::Tz;

use crate::models::{day_start_plus_days, Attempt, Skill, DEFAULT_EASE, MIN_EASE};

/// The lowest `q` that counts as a pass. Below it the skill lapses: reps reset, the
/// interval drops to one day, and the lapse count goes up.
pub const PASSING_QUALITY: u8 = 3;

/// The top of the quality scale, and the value a perfect session produces.
pub const MAX_QUALITY: u8 = 5;

/// The first two intervals of the ladder are fixed rather than computed — SM-2's
/// one piece of hard-coded curve, and the reason a newly learned skill comes back
/// tomorrow and then next week instead of on some ease-derived fraction of a day.
pub const FIRST_INTERVAL_DAYS: u32 = 1;
pub const SECOND_INTERVAL_DAYS: u32 = 6;

/// Where a lapse puts the skill: tomorrow, from scratch.
pub const LAPSE_INTERVAL_DAYS: u32 = 1;

/// Ceiling on a computed interval, in days (ten years).
///
/// Not in the classic algorithm, and it is here for a mechanical reason rather than
/// a pedagogical one: [`day_start_plus_days`] falls back to the *input instant*
/// when the date arithmetic overflows its range, so an interval large enough to run
/// off the end of the calendar would set `due_at` to **now** — the skill becomes due
/// immediately, which is the exact inverse of what a 400-year interval meant. The
/// ladder is `ceil(interval × ease)` with `ease ≥ 1.3`, so it compounds; a row whose
/// `interval_days` was hand-edited (or that survives a long enough streak) gets
/// there. Ten years is indistinguishable from "never" for a learner and stays
/// comfortably inside the calendar.
pub const MAX_INTERVAL_DAYS: u32 = 3650;

/// The SM-2 half of a skill's state — the half [`review`] moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    pub ease: f64,
    pub interval_days: u32,
    pub reps: u32,
    pub lapses: u32,
}

impl Default for Schedule {
    /// A never-reviewed skill: default ease, no interval, no history.
    fn default() -> Self {
        Self {
            ease: DEFAULT_EASE,
            interval_days: 0,
            reps: 0,
            lapses: 0,
        }
    }
}

impl Schedule {
    pub fn of(skill: &Skill) -> Self {
        Self {
            ease: skill.ease,
            interval_days: skill.interval_days,
            reps: skill.reps,
            lapses: skill.lapses,
        }
    }
}

/// What [`review`] produced: the new state and the instant it is next owed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Review {
    pub schedule: Schedule,
    /// A day boundary in the subject's timezone, ready for
    /// [`crate::store::TuitionStore::update_skill_schedule`].
    pub due_at: i64,
    /// `true` when `q` was below [`PASSING_QUALITY`]. Surfaced because a lapse is
    /// what the `mastery.dropped` event is about, and recomputing "did that count
    /// as a lapse" from the before/after state is guesswork.
    pub lapsed: bool,
}

/// One graded answer, reduced to what the quality grade needs.
///
/// Deliberately not `&Attempt`: [`quality`] is also fed by the `tuition__grade` MCP
/// tool and by tests, neither of which has a persisted attempt row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// `None` while a free-response answer is still awaiting its rubric mark.
    pub correct: Option<bool>,
    /// Partial credit in `[0,1]` where the kind supports it.
    pub score: Option<f64>,
}

impl Outcome {
    /// Read an outcome off a stored attempt.
    ///
    /// Note what is **not** consulted: `informative`. That flag is a statement about
    /// the BKT denominator — whether the posterior had anything to condition on —
    /// and not about whether the answer was observed. A wrong answer under
    /// degenerate parameters is still a wrong answer, and dropping it here would
    /// mean a session the learner failed outright produced no `q` at all, so the
    /// skill would not lapse and would not be rescheduled: the sitting would leave
    /// no trace. `informative` *is* honoured in [`crate::trajectory`], where the
    /// quantity being averaged is a posterior movement and a non-movement would
    /// drag the observed learning rate toward zero.
    pub fn of(attempt: &Attempt) -> Self {
        Self {
            correct: attempt.correct,
            score: attempt.score,
        }
    }
}

/// The share of the session that was right, in `[0,1]`.
///
/// Each graded attempt contributes its `score` when it has one (a cloze answered
/// three blanks out of four is 0.75, not a flat failure) and 1.0/0.0 from `correct`
/// otherwise. Ungraded attempts — a free response still waiting on its rubric —
/// contribute nothing and are not counted in the denominator either; grading them
/// later and re-running this is what moves the schedule.
///
/// `None` when nothing in the slice was graded: there is no ratio to take, and the
/// caller must leave the schedule alone rather than treat "no evidence" as zero.
pub fn weighted_correct_ratio(outcomes: &[Outcome]) -> Option<f64> {
    let mut total = 0.0f64;
    let mut counted = 0usize;
    for outcome in outcomes {
        let Some(correct) = outcome.correct else {
            continue;
        };
        let credit = match outcome.score {
            Some(score) if score.is_finite() => score.clamp(0.0, 1.0),
            _ => {
                if correct {
                    1.0
                } else {
                    0.0
                }
            }
        };
        total += credit;
        counted += 1;
    }
    if counted == 0 {
        return None;
    }
    Some(total / counted as f64)
}

/// `q = round(5 × weighted_correct_ratio)`, in `0..=5`.
pub fn quality(outcomes: &[Outcome]) -> Option<u8> {
    let ratio = weighted_correct_ratio(outcomes)?;
    let q = (f64::from(MAX_QUALITY) * ratio).round();
    Some(q.clamp(0.0, f64::from(MAX_QUALITY)) as u8)
}

/// The SM-2 step.
///
/// ```text
/// q < 3:  reps = 0; interval = 1; lapses += 1
/// q >= 3: reps += 1
///         interval = match reps { 1 => 1, 2 => 6, _ => ceil(interval × ease) }
///         ease = max(1.3, ease + (0.1 - (5-q)(0.08 + (5-q)·0.02)))
/// due_at = day_start(now) + interval days
/// ```
///
/// Two details are load-bearing and both are as SM-2 defines them, not as a reader
/// might tidy them:
///
/// - The **interval is computed with the ease from before this review**, and the
///   ease is updated afterwards. Swapping the two applies a penalty to the very
///   review that earned it and makes the ladder drop a step early.
/// - A **lapse does not touch the ease** in this variant. The reset to one day is
///   the whole penalty; also decaying the ease double-counts one bad sitting, and
///   two of them then pin the skill at the [`MIN_EASE`] floor for good.
pub fn review(prev: Schedule, q: u8, tz: Tz, now: i64) -> Review {
    let q = q.min(MAX_QUALITY);
    let mut next = prev;
    // A hand-edited or corrupt ease is floored here as well as at the store's write
    // path: an ease below 1.3 collapses every later interval to nothing and the
    // skill reappears every single day forever.
    next.ease = if prev.ease.is_finite() {
        prev.ease.max(MIN_EASE)
    } else {
        DEFAULT_EASE
    };

    let lapsed = q < PASSING_QUALITY;
    if lapsed {
        next.reps = 0;
        next.interval_days = LAPSE_INTERVAL_DAYS;
        next.lapses = prev.lapses.saturating_add(1);
    } else {
        next.reps = prev.reps.saturating_add(1);
        next.interval_days = match next.reps {
            1 => FIRST_INTERVAL_DAYS,
            2 => SECOND_INTERVAL_DAYS,
            // `.max(1)` covers the row that arrives with reps ≥ 2 and a zero
            // interval — only reachable by hand-editing, but `ceil(0 × ease) == 0`
            // would schedule the skill for today, every day, with no way out.
            _ => scale_interval(prev.interval_days.max(1), next.ease),
        };
        let miss = f64::from(MAX_QUALITY - q);
        next.ease = MIN_EASE.max(next.ease + (0.1 - miss * (0.08 + miss * 0.02)));
    }
    next.interval_days = next.interval_days.clamp(1, MAX_INTERVAL_DAYS);

    Review {
        schedule: next,
        due_at: day_start_plus_days(tz, now, next.interval_days),
        lapsed,
    }
}

/// `ceil(interval × ease)`, saturating at [`MAX_INTERVAL_DAYS`].
///
/// The multiplication is the one place this module leaves integers, and it cannot
/// avoid it — the ease is a real number by construction. It is immediately brought
/// back: the `f64` product is ceilinged and clamped into `u32` range before it can
/// reach anything that stores it, so no downstream consumer ever sees a fractional
/// or out-of-range day count.
fn scale_interval(interval_days: u32, ease: f64) -> u32 {
    let scaled = (f64::from(interval_days) * ease).ceil();
    if !scaled.is_finite() || scaled <= 0.0 {
        return FIRST_INTERVAL_DAYS;
    }
    if scaled >= f64::from(MAX_INTERVAL_DAYS) {
        return MAX_INTERVAL_DAYS;
    }
    scaled as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{day_start_ms, resolve_tz};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    struct QualityCase {
        name: &'static str,
        outcomes: &'static [(Option<bool>, Option<f64>)],
        expect: Option<u8>,
    }

    #[test]
    fn the_quality_grade_is_derived_from_what_was_answered() {
        let cases = [
            QualityCase {
                name: "a clean sweep is a 5",
                outcomes: &[(Some(true), None), (Some(true), None), (Some(true), None)],
                expect: Some(5),
            },
            QualityCase {
                name: "everything wrong is a 0, which lapses the skill",
                outcomes: &[(Some(false), None), (Some(false), None)],
                expect: Some(0),
            },
            QualityCase {
                name: "half right rounds to the middle of the scale",
                outcomes: &[(Some(true), None), (Some(false), None)],
                expect: Some(3),
            },
            QualityCase {
                name: "partial credit counts as partial, not as a failure",
                outcomes: &[(Some(false), Some(0.75)), (Some(true), Some(1.0))],
                // (0.75 + 1.0) / 2 = 0.875 → round(4.375) = 4
                expect: Some(4),
            },
            QualityCase {
                name: "an ungraded free response contributes nothing, either way",
                outcomes: &[(Some(true), None), (None, None)],
                expect: Some(5),
            },
            QualityCase {
                name: "a session with nothing graded yet yields no grade at all",
                outcomes: &[(None, None), (None, None)],
                expect: None,
            },
            QualityCase {
                name: "an empty session yields no grade",
                outcomes: &[],
                expect: None,
            },
            QualityCase {
                name: "a corrupt score falls back to the pass/fail flag",
                outcomes: &[(Some(true), Some(f64::NAN))],
                expect: Some(5),
            },
        ];
        for case in cases {
            let outcomes: Vec<Outcome> = case
                .outcomes
                .iter()
                .map(|(correct, score)| Outcome {
                    correct: *correct,
                    score: *score,
                })
                .collect();
            assert_eq!(quality(&outcomes), case.expect, "{}", case.name);
        }
    }

    #[test]
    fn the_interval_ladder_climbs_one_two_six_then_by_ease() {
        let tz = resolve_tz("UTC");
        let now = 1_786_000_000_000;
        let mut state = Schedule::default();

        state = review(state, 5, tz, now).schedule;
        assert_eq!((state.reps, state.interval_days), (1, 1));
        state = review(state, 5, tz, now).schedule;
        assert_eq!((state.reps, state.interval_days), (2, 6));
        // Third review: ceil(6 × 2.7) — the ease used is the one from BEFORE this
        // review's own bump, i.e. 2.5 raised twice by the two perfect sittings.
        let third = review(state, 5, tz, now);
        assert_eq!(third.schedule.reps, 3);
        assert_eq!(
            third.schedule.interval_days,
            (6.0f64 * state.ease).ceil() as u32
        );
        assert!(third.schedule.interval_days > 6);
    }

    #[test]
    fn ease_moves_by_the_sm2_formula_and_never_below_the_floor() {
        let tz = resolve_tz("UTC");
        let now = 1_786_000_000_000;
        // q = 5 → +0.1; q = 4 → -0.0 exactly; q = 3 → -0.14.
        let table: &[(u8, f64)] = &[(5, 0.1), (4, 0.0), (3, -0.14)];
        for (q, delta) in table {
            let got = review(Schedule::default(), *q, tz, now).schedule.ease;
            assert!(
                close(got, DEFAULT_EASE + delta),
                "q={q}: expected {}, got {got}",
                DEFAULT_EASE + delta
            );
        }
        // Repeated barely-passing reviews walk the ease down to the floor and stop.
        let mut state = Schedule::default();
        for _ in 0..40 {
            state = review(state, 3, tz, 1_786_000_000_000).schedule;
        }
        assert!(close(state.ease, MIN_EASE));
    }

    #[test]
    fn a_failed_review_lapses_to_tomorrow_without_also_decaying_the_ease() {
        let tz = resolve_tz("UTC");
        let now = 1_786_000_000_000;
        let studied = Schedule {
            ease: 2.6,
            interval_days: 30,
            reps: 5,
            lapses: 0,
        };
        let got = review(studied, 2, tz, now);
        assert!(got.lapsed);
        assert_eq!(got.schedule.reps, 0);
        assert_eq!(got.schedule.interval_days, LAPSE_INTERVAL_DAYS);
        assert_eq!(got.schedule.lapses, 1);
        // The reset to one day IS the penalty; decaying the ease too would
        // double-count one bad sitting.
        assert!(close(got.schedule.ease, 2.6));
    }

    #[test]
    fn a_session_at_2358_schedules_the_same_day_as_one_at_0002_the_next_morning() {
        // THE 23:58 SESSION. `due_at` is a day boundary, so a late-night sitting and
        // an early-morning one on the following day both land on "tomorrow" relative
        // to their own local day — the review does not creep later every night.
        let tz = resolve_tz("Asia/Tokyo");
        let late = 1_786_373_880_000; // 2026-08-10T14:58:00Z == 23:58 in Tokyo
        let early = late + 4 * 60_000; // 2026-08-11T00:02 local, the NEXT local day

        let after_late = review(Schedule::default(), 5, tz, late);
        let after_early = review(Schedule::default(), 5, tz, early);

        // Each one lands exactly on a boundary…
        assert_eq!(after_late.due_at, day_start_ms(tz, after_late.due_at));
        assert_eq!(after_early.due_at, day_start_ms(tz, after_early.due_at));
        // …one whole local day after the day the session belonged to…
        assert_eq!(after_late.due_at - day_start_ms(tz, late), 24 * 3_600_000);
        // …and the two differ by exactly one day, not by the four minutes between
        // the sessions, which is what `now + 86_400_000` would have produced.
        assert_eq!(after_early.due_at - after_late.due_at, 24 * 3_600_000);
    }

    #[test]
    fn an_absurd_stored_interval_still_produces_a_future_due_date() {
        // A hand-edited row (or an implausibly long streak) must not run off the end
        // of the calendar: `day_start_plus_days` returns the INPUT instant when the
        // date arithmetic overflows, which would make the skill due immediately —
        // silently, and exactly backwards.
        let tz = resolve_tz("Europe/Berlin");
        let now = 1_786_000_000_000;
        let absurd = Schedule {
            ease: 2.5,
            interval_days: u32::MAX,
            reps: 9,
            lapses: 0,
        };
        let got = review(absurd, 5, tz, now);
        assert_eq!(got.schedule.interval_days, MAX_INTERVAL_DAYS);
        assert!(got.due_at > now);
        assert_eq!(got.due_at, day_start_ms(tz, got.due_at));
    }

    #[test]
    fn a_corrupt_ease_is_repaired_rather_than_propagated() {
        let tz = resolve_tz("UTC");
        let now = 1_786_000_000_000;
        let corrupt = Schedule {
            ease: f64::NAN,
            interval_days: 10,
            reps: 4,
            lapses: 0,
        };
        let got = review(corrupt, 5, tz, now).schedule;
        assert!(got.ease.is_finite());
        assert!(got.ease >= MIN_EASE);
        assert!(got.interval_days >= 1);

        let below_floor = Schedule {
            ease: 0.2,
            ..corrupt
        };
        let got = review(below_floor, 3, tz, now).schedule;
        assert!(close(got.ease, MIN_EASE));
    }
}
