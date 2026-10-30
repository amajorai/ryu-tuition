//! Bayesian Knowledge Tracing — the posterior behind every mastery number this app
//! shows.
//!
//! One attempt, one Bayes step. Given the prior `P(known)` and whether the answer
//! was right, [`update`] produces the posterior and then applies the learning
//! transition — the learner may have *acquired* the skill on this attempt, which is
//! what separates BKT from a running accuracy average and what makes "0.62 mastery"
//! mean something after three attempts rather than thirty.
//!
//! Everything here is a pure function of numbers passed in. There is no clock read
//! and no store access, deliberately: a posterior that depends on when it was
//! computed cannot be audited, and auditability is the entire claim this app makes.
//!
//! # The two traps
//!
//! **The guess rate is per item, not per skill.** A 2-choice question is right half
//! the time from a coin flip; a 4-choice one a quarter of the time. Grading the
//! first with the second's number over-credits every correct answer, and the
//! mastery curve then climbs on questions the learner was guessing at. So the
//! stored `p_guess` is a floor-adjusted input, not the value used directly — see
//! [`guess_rate`], and note that the forward projection takes the *same already
//! floored* rate rather than re-deriving it, so "3 more correct reaches 0.9" is a
//! statement about the questions the learner will actually be asked.
//!
//! **A degenerate denominator is not an error, it is an uninformative attempt.**
//! With `p_slip = 0`, a wrong answer from a fully-mastered skill has probability
//! zero: the model says it cannot happen, and Bayes has nothing to divide by. The
//! naive implementation produces `NaN`, which then propagates into every later
//! update and silently blanks the whole mastery model. Here the posterior is left
//! exactly where it was and the attempt is flagged `informative = false`, which is
//! also what [`crate::models::Attempt::informative`] records — so "why did nothing
//! move" has an answer on the row itself.

use crate::models::{BktParams, ItemKind};

/// Denominators at or below this carry no information: the model assigned the
/// observed outcome probability zero, so there is nothing to condition on.
///
/// Not `== 0.0`. The denominator is a sum of products of clamped probabilities, so
/// a parameter set that is *effectively* degenerate (`p_slip = 1e-300`) lands a
/// hair above zero and would produce a posterior of the form `tiny / tiny` — a
/// number with no meaning that is nevertheless finite and would be stored.
pub const DENOMINATOR_EPSILON: f64 = 1e-12;

/// Ceiling on the forward projection's iterations.
///
/// It is a projection, not a promise, and past twenty consecutive correct answers
/// the premise ("at this difficulty, answering correctly every time") has stopped
/// describing anything a learner will actually do. Reporting `None` — rendered as
/// "not within 20" — is more honest than a number like 340.
pub const MAX_PROJECTION_STEPS: u32 = 20;

/// Clamp into `[0,1]`, mapping non-finite input to 0.
///
/// Applied to every input rather than assumed: `mastery` arrives from a database
/// column that a `sqlite3` shell can write, and one out-of-range prior would make
/// every posterior derived from it meaningless.
fn clamp01(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The outcome of one BKT step, carrying both endpoints.
///
/// Both are returned rather than just the new value because
/// [`crate::models::Attempt`] stores `mastery_before` and `mastery_after` on the
/// attempt row — a posterior you cannot walk backwards is a posterior you cannot
/// check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BktUpdate {
    pub before: f64,
    pub after: f64,
    /// `false` when the denominator was degenerate and `after == before`. The
    /// attempt still happened and is still recorded; it just moved nothing.
    pub informative: bool,
}

/// The guess rate to grade one item with: the skill's `p_guess`, floored by the
/// structural floor of the item's own format.
///
/// An `n`-choice multiple-choice question can be answered correctly by chance with
/// probability `1/n`, so the effective guess rate is `max(p_guess, 1/n)`. The other
/// kinds have no such floor — there is no `1/n` for "type the answer" — so they use
/// the stored parameter unchanged.
///
/// `n < 2` is deliberately *not* floored. A one-choice (or zero-choice) MCQ is a
/// malformed item, and `1/1 = 1.0` would make every wrong answer's denominator
/// degenerate, quietly turning all of that item's attempts non-informative. Leaving
/// the stored rate alone keeps the malformed item gradeable and visible rather than
/// silently inert.
pub fn guess_rate(params: BktParams, kind: ItemKind, choice_count: usize) -> f64 {
    let base = clamp01(params.clamped().p_guess);
    match kind {
        ItemKind::Mcq if choice_count >= 2 => base.max(1.0 / choice_count as f64),
        _ => base,
    }
}

/// One Bayes step plus the learning transition.
///
/// `guess` is the *effective* rate from [`guess_rate`], not `params.p_guess` — the
/// caller has already decided what the item's format implies, and re-deriving it
/// here would need the item, which this module deliberately does not take.
///
/// ```text
/// correct:   post = m(1-slip) / [ m(1-slip) + (1-m)·guess ]
/// incorrect: post = m·slip    / [ m·slip    + (1-m)(1-guess) ]
/// m'        = post + (1 - post)·transit
/// ```
pub fn update(mastery: f64, params: BktParams, guess: f64, correct: bool) -> BktUpdate {
    let params = params.clamped();
    let m = clamp01(mastery);
    let guess = clamp01(guess);

    let (numerator, denominator) = if correct {
        let known = m * (1.0 - params.p_slip);
        (known, known + (1.0 - m) * guess)
    } else {
        let known = m * params.p_slip;
        (known, known + (1.0 - m) * (1.0 - guess))
    };

    if !denominator.is_finite() || denominator <= DENOMINATOR_EPSILON {
        return BktUpdate {
            before: m,
            after: m,
            informative: false,
        };
    }

    let post = numerator / denominator;
    // The division is guarded but the result still gets checked: a finite
    // numerator over a finite denominator can only be non-finite through a bug,
    // and a bug must not be allowed to write NaN into the column every later
    // attempt reads as its prior.
    if !post.is_finite() {
        return BktUpdate {
            before: m,
            after: m,
            informative: false,
        };
    }
    let after = clamp01(post + (1.0 - post) * params.p_transit);
    BktUpdate {
        before: m,
        after,
        informative: true,
    }
}

/// [`update`] with the item's format folded in — the form the session runner uses.
pub fn update_for_item(
    mastery: f64,
    params: BktParams,
    kind: ItemKind,
    choice_count: usize,
    correct: bool,
) -> BktUpdate {
    update(
        mastery,
        params,
        guess_rate(params, kind, choice_count),
        correct,
    )
}

/// "`k` more correct at this difficulty reaches `target`."
///
/// A projection, and the type says so: `steps` is `None` when the target is not
/// reachable within [`MAX_PROJECTION_STEPS`], and `reached` then says how far
/// twenty correct answers would actually get — which is the sentence a learner
/// needs ("twenty more only takes you to 0.71" means the parameters are wrong, not
/// that they should keep grinding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    pub from: f64,
    pub target: f64,
    /// Consecutive correct answers needed. `Some(0)` when already at target.
    pub steps: Option<u32>,
    /// The posterior after `steps` (or after the cap, when `steps` is `None`).
    pub reached: f64,
}

/// Iterate [`update`] forward with `correct = true` until the posterior reaches
/// `target`, capped at [`MAX_PROJECTION_STEPS`].
///
/// Takes the already-floored `guess` for the same reason [`update`] does, and it is
/// the same value: projecting with the raw `p_guess` while grading with the MCQ
/// floor would make the projection describe a question the learner is never asked.
///
/// A non-informative step ends the walk immediately — the posterior has stopped
/// moving, so no number of further attempts changes it, and iterating to the cap
/// would only burn the loop to arrive at the same answer.
pub fn correct_answers_to_target(
    mastery: f64,
    params: BktParams,
    guess: f64,
    target: f64,
) -> Projection {
    let from = clamp01(mastery);
    let target = clamp01(target);
    let mut m = from;
    if m >= target {
        return Projection {
            from,
            target,
            steps: Some(0),
            reached: m,
        };
    }
    for step in 1..=MAX_PROJECTION_STEPS {
        let next = update(m, params, guess, true);
        if !next.informative {
            break;
        }
        m = next.after;
        if m >= target {
            return Projection {
                from,
                target,
                steps: Some(step),
                reached: m,
            };
        }
    }
    Projection {
        from,
        target,
        steps: None,
        reached: m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Posteriors are compared with an epsilon, never `==`: every one of them is
    /// the result of a division, and an exact comparison here would be a test that
    /// passes or fails on the last bit of a mantissa.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn params(p_init: f64, p_transit: f64, p_slip: f64, p_guess: f64) -> BktParams {
        BktParams {
            p_init,
            p_transit,
            p_slip,
            p_guess,
        }
    }

    struct UpdateCase {
        name: &'static str,
        mastery: f64,
        params: BktParams,
        guess: f64,
        correct: bool,
        expect_after: f64,
        expect_informative: bool,
    }

    #[test]
    fn the_posterior_update_matches_hand_computed_bayes() {
        // Every expectation below is worked by hand from the two formulas in the
        // module docs, so this table is a check on the code and not a recording of
        // whatever it happened to print.
        let default = params(0.20, 0.15, 0.10, 0.20);
        let cases = [
            UpdateCase {
                name: "correct answer raises the posterior",
                mastery: 0.5,
                params: default,
                guess: 0.20,
                correct: true,
                // post = .5*.9 / (.5*.9 + .5*.2) = .45/.55 = 0.818181…
                // m'   = post + (1-post)*.15 = 0.845454…
                expect_after: 0.45 / 0.55 + (1.0 - 0.45 / 0.55) * 0.15,
                expect_informative: true,
            },
            UpdateCase {
                name: "wrong answer lowers it, but the transition still lifts it a little",
                mastery: 0.5,
                params: default,
                guess: 0.20,
                correct: false,
                // post = .5*.1 / (.5*.1 + .5*.8) = .05/.45 = 0.111111…
                expect_after: 0.05 / 0.45 + (1.0 - 0.05 / 0.45) * 0.15,
                expect_informative: true,
            },
            UpdateCase {
                name: "a certain-knowledge prior is not moved by a correct answer",
                mastery: 1.0,
                params: default,
                guess: 0.20,
                correct: true,
                expect_after: 1.0,
                expect_informative: true,
            },
            UpdateCase {
                // THE DEGENERATE DENOMINATOR. p_slip = 0 says a known skill is never
                // answered wrong, so this outcome has probability zero and Bayes has
                // nothing to divide by. The naive implementation writes NaN here and
                // every later attempt inherits it.
                name: "impossible outcome leaves the posterior alone instead of producing NaN",
                mastery: 1.0,
                params: params(0.20, 0.15, 0.0, 0.20),
                guess: 0.20,
                correct: false,
                expect_after: 1.0,
                expect_informative: false,
            },
            UpdateCase {
                name: "an unguessable item answered right from a zero prior is also degenerate",
                mastery: 0.0,
                params: params(0.0, 0.15, 0.10, 0.0),
                guess: 0.0,
                correct: true,
                expect_after: 0.0,
                expect_informative: false,
            },
            UpdateCase {
                name: "a hand-edited out-of-range prior is clamped, not propagated",
                mastery: 4.2,
                params: default,
                guess: 0.20,
                correct: true,
                expect_after: 1.0,
                expect_informative: true,
            },
        ];

        for case in cases {
            let got = update(case.mastery, case.params, case.guess, case.correct);
            assert!(
                close(got.after, case.expect_after),
                "{}: expected {}, got {}",
                case.name,
                case.expect_after,
                got.after
            );
            assert_eq!(got.informative, case.expect_informative, "{}", case.name);
            assert!(
                got.after.is_finite(),
                "{}: produced a non-finite posterior",
                case.name
            );
        }
    }

    #[test]
    fn the_guess_rate_floor_is_per_item_kind() {
        let p = params(0.20, 0.15, 0.10, 0.20);
        let cases: &[(&str, ItemKind, usize, f64)] = &[
            // THE TWO-CHOICE MCQ: a coin flip is right half the time and the stored
            // 0.20 does not describe it.
            ("two-choice mcq floors at 1/2", ItemKind::Mcq, 2, 0.5),
            ("four-choice mcq floors at 1/4", ItemKind::Mcq, 4, 0.25),
            (
                "five-choice mcq is exactly the stored rate",
                ItemKind::Mcq,
                5,
                0.20,
            ),
            (
                "ten-choice mcq keeps the stored rate",
                ItemKind::Mcq,
                10,
                0.20,
            ),
            // A one-choice MCQ is malformed. Flooring at 1/1 would make every wrong
            // answer degenerate and quietly stop the item from teaching anything.
            (
                "one-choice mcq is not floored to certainty",
                ItemKind::Mcq,
                1,
                0.20,
            ),
            (
                "zero-choice mcq is not floored either",
                ItemKind::Mcq,
                0,
                0.20,
            ),
            ("cloze has no structural floor", ItemKind::Cloze, 4, 0.20),
            (
                "numeric has no structural floor",
                ItemKind::Numeric,
                0,
                0.20,
            ),
            (
                "free response has no structural floor",
                ItemKind::Free,
                0,
                0.20,
            ),
        ];
        for (name, kind, choices, expect) in cases {
            let got = guess_rate(p, *kind, *choices);
            assert!(close(got, *expect), "{name}: expected {expect}, got {got}");
        }
    }

    #[test]
    fn grading_a_two_choice_item_at_the_four_choice_rate_over_credits_it() {
        // The lie the floor exists to prevent, stated as an inequality: a correct
        // coin-flip answer must move the posterior LESS than a correct 4-choice one,
        // because it is weaker evidence. Without the floor both use 0.20 and the
        // model treats them as the same evidence.
        let p = params(0.20, 0.15, 0.10, 0.20);
        let two = update(0.5, p, guess_rate(p, ItemKind::Mcq, 2), true);
        let four = update(0.5, p, guess_rate(p, ItemKind::Mcq, 4), true);
        let unfloored = update(0.5, p, p.p_guess, true);
        assert!(two.after < four.after);
        assert!(four.after < unfloored.after);
    }

    #[test]
    fn the_projection_agrees_with_actually_applying_the_updates() {
        // The projection is the one number a learner is invited to plan against
        // ("three more and you're there"), so it has to be the same arithmetic the
        // session will run — same params, same floored guess rate.
        let p = params(0.20, 0.15, 0.10, 0.20);
        let guess = guess_rate(p, ItemKind::Mcq, 2);
        let projection = correct_answers_to_target(0.2, p, guess, 0.9);
        let steps = projection.steps.expect("0.9 is reachable from 0.2");

        let mut m = 0.2;
        for _ in 0..steps {
            m = update(m, p, guess, true).after;
        }
        assert!(m >= 0.9, "replaying {steps} updates must reach the target");
        assert!(close(m, projection.reached));

        // …and one step short must NOT reach it, or the projection is over-counting.
        let mut short = 0.2;
        for _ in 0..steps.saturating_sub(1) {
            short = update(short, p, guess, true).after;
        }
        assert!(short < 0.9);
    }

    #[test]
    fn the_projection_reports_unreachable_rather_than_a_number_nobody_would_act_on() {
        // No learning transition and a heavy guess rate: the posterior creeps and
        // never arrives. Twenty is the cap, and what comes back is where twenty
        // correct answers actually land.
        let p = params(0.20, 0.0, 0.45, 0.45);
        let projection = correct_answers_to_target(0.2, p, 0.45, 0.99);
        assert_eq!(projection.steps, None);
        assert!(projection.reached < 0.99);
        assert!(projection.reached.is_finite());
        assert!(projection.reached > projection.from);
    }

    #[test]
    fn a_degenerate_step_ends_the_projection_instead_of_spinning_to_the_cap() {
        // A zero prior on an unguessable item: the first step's denominator is zero,
        // so no number of further correct answers moves anything. The walk must stop
        // at the first non-informative step rather than burn twenty of them to
        // arrive at the same "unreachable".
        let p = params(0.0, 0.15, 0.10, 0.0);
        let projection = correct_answers_to_target(0.0, p, 0.0, 0.9);
        assert_eq!(projection.steps, None);
        assert!(close(projection.reached, 0.0));
    }

    #[test]
    fn already_at_target_needs_no_further_answers() {
        let p = params(0.20, 0.15, 0.10, 0.20);
        let projection = correct_answers_to_target(0.95, p, 0.20, 0.9);
        assert_eq!(projection.steps, Some(0));
        assert!(close(projection.reached, 0.95));
    }
}
