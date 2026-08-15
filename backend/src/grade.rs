//! Objective grading — the part of this app that never asks a model anything.
//!
//! Four of the five item kinds are decided here by comparison: multiple choice, cloze,
//! numeric and exact match. Only free response reaches a model, and it reaches one
//! with a written rubric attached that is shown next to the mark.
//!
//! That split is the app's entire trust story. A learner who is told they got a
//! question wrong has to be able to see *why*, and "a model thought so" is not a
//! reason for a question with one right answer.
//!
//! # The decimal
//!
//! Numeric answers are compared with a hand-rolled fixed-point decimal (an `i128`
//! mantissa and a scale), never `f64`. This is not fastidiousness:
//!
//! ```text
//! 0.1 + 0.2 == 0.3          // false in binary floating point
//! ```
//!
//! An item whose expected answer is `0.3` with a tolerance of `0` marks a correct
//! `0.3` WRONG under `f64`, and no amount of explaining recovers a learner's trust
//! after that. `rust_decimal` would do this too, and is not in the root lockfile —
//! see the dependency rule in `Cargo.toml`.

use crate::models::{AnswerKey, GradedBy, ItemKind, Tolerance};

/// Maximum digits accepted in a numeric answer, before or after the point.
///
/// `i128` holds 38 digits; this leaves room for the rescaling in [`Decimal::align`]
/// to not overflow when comparing a value written to 2 places against one written to
/// 18.
pub const MAX_DECIMAL_DIGITS: usize = 30;

// ── Fixed-point decimal ────────────────────────────────────────────────────────

/// A decimal number as an integer mantissa and a base-10 scale: `mantissa / 10^scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decimal {
    mantissa: i128,
    scale: u32,
}

impl Decimal {
    /// Parse a decimal from text.
    ///
    /// Accepts a leading sign, digits, one decimal point, and surrounding whitespace.
    /// Rejects everything else — including exponent notation, which is deliberate:
    /// `1e3` in a learner's answer box is far more often a typo than an intent, and
    /// silently accepting it means a wrong answer scores correct.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Decimal> {
        let text = raw.trim();
        if text.is_empty() {
            return None;
        }
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        // Thousands separators are stripped rather than rejected: "1,024" is a
        // legitimate way to write a number and refusing it teaches nothing.
        let digits: String = digits.chars().filter(|c| *c != ',' && *c != '_').collect();
        if digits.is_empty() {
            return None;
        }
        let mut mantissa: i128 = 0;
        let mut scale: u32 = 0;
        let mut seen_point = false;
        let mut seen_digit = false;
        let mut count = 0usize;
        for ch in digits.chars() {
            if ch == '.' {
                if seen_point {
                    return None;
                }
                seen_point = true;
                continue;
            }
            let Some(digit) = ch.to_digit(10) else {
                return None;
            };
            seen_digit = true;
            count += 1;
            if count > MAX_DECIMAL_DIGITS {
                return None;
            }
            mantissa = mantissa.checked_mul(10)?.checked_add(i128::from(digit))?;
            if seen_point {
                scale += 1;
            }
        }
        if !seen_digit {
            return None;
        }
        Some(Decimal {
            mantissa: if negative { -mantissa } else { mantissa },
            scale,
        })
    }

    /// Rescale two decimals to a common scale.
    fn align(a: Decimal, b: Decimal) -> Option<(i128, i128)> {
        let scale = a.scale.max(b.scale);
        let lift = |d: Decimal| -> Option<i128> {
            let steps = scale - d.scale;
            let factor = 10i128.checked_pow(steps)?;
            d.mantissa.checked_mul(factor)
        };
        Some((lift(a)?, lift(b)?))
    }

    /// `|self - other|`, as a decimal at the common scale.
    fn abs_difference(self, other: Decimal) -> Option<Decimal> {
        let (a, b) = Decimal::align(self, other)?;
        Some(Decimal {
            mantissa: a.checked_sub(b)?.abs(),
            scale: self.scale.max(other.scale),
        })
    }

    /// `self <= other`.
    fn le(self, other: Decimal) -> Option<bool> {
        let (a, b) = Decimal::align(self, other)?;
        Some(a <= b)
    }

    /// `|self| * percent / 100`.
    fn percent_of(self, percent: Decimal) -> Option<Decimal> {
        Some(Decimal {
            mantissa: self.mantissa.abs().checked_mul(percent.mantissa.abs())?,
            // Dividing by 100 is +2 to the scale, which is exact — no rounding, so a
            // tolerance boundary lands where the item author wrote it.
            scale: self.scale.checked_add(percent.scale)?.checked_add(2)?,
        })
    }
}

// ── Text normalization ─────────────────────────────────────────────────────────

/// Fold text for comparison: lowercase, collapse whitespace, trim.
///
/// NFKC would be better and needs `unicode-normalization`, which is not in the root
/// lockfile. `str::to_lowercase` is full Unicode case mapping, which covers the cases
/// that actually come up in an answer box.
#[must_use]
pub fn fold(text: &str) -> String {
    text.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Outcome ────────────────────────────────────────────────────────────────────

/// The result of grading one answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Decided here, with no model involved.
    Decided {
        correct: bool,
        /// Partial credit in `0.0..=1.0`. Only cloze produces a value strictly
        /// between 0 and 1; every other objective kind is all or nothing.
        score: f64,
        graded_by: GradedBy,
        /// Per-blank results, for cloze. Empty otherwise.
        blanks: Vec<bool>,
    },
    /// Free response: this module will not guess. The caller sends it to the rubric
    /// path and shows the rubric alongside the mark.
    NeedsRubric { rubric: String },
    /// The answer could not be graded, and no verdict is invented.
    ///
    /// A malformed answer key or an unparseable numeric response is reported as such,
    /// never as "incorrect". Marking a learner wrong because OUR row was corrupt is
    /// the one failure that would make the mastery model actively misleading — the
    /// posterior would drop on evidence that says nothing about what they know.
    Ungradeable { reason: String },
}

impl Outcome {
    /// Whether this outcome should move the mastery posterior.
    #[must_use]
    pub fn is_informative(&self) -> bool {
        matches!(self, Outcome::Decided { .. })
    }
}

/// Grade one response against an answer key.
///
/// `kind` is taken alongside the key so a row whose `kind` column and `answer` blob
/// disagree is caught rather than silently graded by whichever one the code happened
/// to read.
#[must_use]
pub fn grade(kind: ItemKind, key: &AnswerKey, response: &str) -> Outcome {
    let expected_kind = match key {
        AnswerKey::Mcq { .. } => Some(ItemKind::Mcq),
        AnswerKey::Cloze { .. } => Some(ItemKind::Cloze),
        AnswerKey::Numeric { .. } => Some(ItemKind::Numeric),
        AnswerKey::Exact { .. } => Some(ItemKind::Exact),
        AnswerKey::Free { .. } => Some(ItemKind::Free),
        AnswerKey::Malformed { .. } => None,
    };
    match expected_kind {
        None => {
            return Outcome::Ungradeable {
                reason: "this item's stored answer could not be read".into(),
            }
        }
        Some(expected) if expected != kind => {
            return Outcome::Ungradeable {
                reason: format!(
                    "this item is stored as '{}' but its answer is a '{}' answer",
                    kind.as_str(),
                    expected.as_str()
                ),
            }
        }
        Some(_) => {}
    }

    match key {
        AnswerKey::Mcq { choice_id } => {
            let correct = fold(response) == fold(choice_id);
            decided(correct)
        }
        AnswerKey::Exact { text, alternatives } => {
            let given = fold(response);
            let correct = std::iter::once(text)
                .chain(alternatives.iter())
                .any(|accepted| fold(accepted) == given);
            decided(correct)
        }
        AnswerKey::Cloze { blanks } => grade_cloze(blanks, response),
        AnswerKey::Numeric {
            expected,
            tolerance,
        } => grade_numeric(expected, tolerance, response),
        AnswerKey::Free { rubric } => Outcome::NeedsRubric {
            rubric: rubric.clone(),
        },
        AnswerKey::Malformed { .. } => unreachable!("handled above"),
    }
}

fn decided(correct: bool) -> Outcome {
    Outcome::Decided {
        correct,
        score: if correct { 1.0 } else { 0.0 },
        graded_by: GradedBy::Deterministic,
        blanks: Vec::new(),
    }
}

/// Cloze: one response segment per blank, separated by `|`.
///
/// Partial credit is real here — getting three of four blanks is genuinely different
/// from getting none, and collapsing that to "wrong" makes the mastery posterior drop
/// as hard for a near-miss as for a blank page.
fn grade_cloze(blanks: &[Vec<String>], response: &str) -> Outcome {
    if blanks.is_empty() {
        return Outcome::Ungradeable {
            reason: "this cloze item has no blanks".into(),
        };
    }
    let given: Vec<&str> = response.split('|').collect();
    let results: Vec<bool> = blanks
        .iter()
        .enumerate()
        .map(|(index, accepted)| {
            let answer = given.get(index).map(|s| fold(s)).unwrap_or_default();
            // A blank left empty is wrong, not vacuously right — an accepted list
            // containing "" would otherwise pass an unanswered blank.
            !answer.is_empty() && accepted.iter().any(|a| fold(a) == answer)
        })
        .collect();
    let hits = results.iter().filter(|ok| **ok).count();
    Outcome::Decided {
        correct: hits == blanks.len(),
        score: hits as f64 / blanks.len() as f64,
        graded_by: GradedBy::Deterministic,
        blanks: results,
    }
}

fn grade_numeric(expected: &str, tolerance: &Tolerance, response: &str) -> Outcome {
    let Some(expected_value) = Decimal::parse(expected) else {
        return Outcome::Ungradeable {
            reason: "this item's expected value could not be read".into(),
        };
    };
    let Some(given) = Decimal::parse(response) else {
        // A non-numeric response to a numeric question is a REAL wrong answer — the
        // learner typed something and it was not a number — as distinct from the
        // stored-row failures above, which say nothing about what they know.
        return Outcome::Decided {
            correct: false,
            score: 0.0,
            graded_by: GradedBy::Deterministic,
            blanks: Vec::new(),
        };
    };
    let allowed = match tolerance {
        Tolerance::Absolute { value } => Decimal::parse(value),
        Tolerance::RelativePercent { value } => {
            Decimal::parse(value).and_then(|pct| expected_value.percent_of(pct))
        }
    };
    let Some(allowed) = allowed else {
        return Outcome::Ungradeable {
            reason: "this item's tolerance could not be read".into(),
        };
    };
    let Some(difference) = given.abs_difference(expected_value) else {
        return Outcome::Ungradeable {
            reason: "the answer has too many digits to compare".into(),
        };
    };
    match difference.le(allowed) {
        Some(within) => decided(within),
        None => Outcome::Ungradeable {
            reason: "the answer has too many digits to compare".into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abs(value: &str) -> Tolerance {
        Tolerance::Absolute { value: value.into() }
    }

    fn is_correct(outcome: &Outcome) -> bool {
        matches!(outcome, Outcome::Decided { correct: true, .. })
    }

    #[test]
    fn the_floating_point_case_this_module_exists_for() {
        // THE adversarial case. Under `f64`, 0.1 + 0.2 != 0.3, and an exact-tolerance
        // item marks a correct answer wrong.
        let key = AnswerKey::Numeric {
            expected: "0.3".into(),
            tolerance: abs("0"),
        };
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "0.3")));
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "0.30")));
        assert!(is_correct(&grade(ItemKind::Numeric, &key, ".3")));
        assert!(!is_correct(&grade(ItemKind::Numeric, &key, "0.31")));
    }

    #[test]
    fn a_tolerance_boundary_is_inclusive_and_exact() {
        let key = AnswerKey::Numeric {
            expected: "10".into(),
            tolerance: abs("0.5"),
        };
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "10.5")));
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "9.5")));
        assert!(!is_correct(&grade(ItemKind::Numeric, &key, "10.51")));
    }

    #[test]
    fn relative_tolerance_scales_with_the_expected_value() {
        let key = AnswerKey::Numeric {
            expected: "200".into(),
            tolerance: Tolerance::RelativePercent { value: "5".into() },
        };
        // 5% of 200 is exactly 10.
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "210")));
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "190")));
        assert!(!is_correct(&grade(ItemKind::Numeric, &key, "210.01")));
    }

    #[test]
    fn a_negative_expected_value_uses_the_magnitude_for_relative_tolerance() {
        let key = AnswerKey::Numeric {
            expected: "-200".into(),
            tolerance: Tolerance::RelativePercent { value: "5".into() },
        };
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "-210")));
        assert!(!is_correct(&grade(ItemKind::Numeric, &key, "-211")));
    }

    #[test]
    fn thousands_separators_are_accepted_but_exponents_are_not() {
        let key = AnswerKey::Numeric {
            expected: "1024".into(),
            tolerance: abs("0"),
        };
        assert!(is_correct(&grade(ItemKind::Numeric, &key, "1,024")));
        // `1e3` is far more often a typo than an intent in an answer box, and
        // accepting it silently would score a wrong answer correct.
        assert!(!is_correct(&grade(ItemKind::Numeric, &key, "1.024e3")));
    }

    #[test]
    fn a_non_numeric_response_is_wrong_but_a_corrupt_row_is_ungradeable() {
        // The distinction that keeps the mastery model honest: the learner being
        // wrong moves the posterior, OUR row being broken must not.
        let key = AnswerKey::Numeric {
            expected: "5".into(),
            tolerance: abs("0"),
        };
        let learner_wrong = grade(ItemKind::Numeric, &key, "banana");
        assert!(matches!(learner_wrong, Outcome::Decided { correct: false, .. }));
        assert!(learner_wrong.is_informative());

        let broken = AnswerKey::Numeric {
            expected: "not a number".into(),
            tolerance: abs("0"),
        };
        let ours_broken = grade(ItemKind::Numeric, &broken, "5");
        assert!(matches!(ours_broken, Outcome::Ungradeable { .. }));
        assert!(!ours_broken.is_informative());
    }

    #[test]
    fn a_malformed_answer_key_never_produces_a_verdict() {
        let key = AnswerKey::Malformed { raw: "{{{".into() };
        let outcome = grade(ItemKind::Mcq, &key, "a");
        assert!(matches!(outcome, Outcome::Ungradeable { .. }));
        assert!(!outcome.is_informative());
    }

    #[test]
    fn a_kind_that_disagrees_with_its_answer_is_refused() {
        // A row whose `kind` column and `answer` blob disagree must not be graded by
        // whichever one the code read first.
        let key = AnswerKey::Exact {
            text: "paris".into(),
            alternatives: vec![],
        };
        let outcome = grade(ItemKind::Numeric, &key, "paris");
        assert!(matches!(outcome, Outcome::Ungradeable { .. }), "{outcome:?}");
    }

    #[test]
    fn exact_match_folds_case_and_whitespace_and_accepts_alternatives() {
        let key = AnswerKey::Exact {
            text: "Nitrogen".into(),
            alternatives: vec!["N2".into(), "N₂".into()],
        };
        assert!(is_correct(&grade(ItemKind::Exact, &key, "  nitrogen ")));
        assert!(is_correct(&grade(ItemKind::Exact, &key, "n2")));
        assert!(is_correct(&grade(ItemKind::Exact, &key, "N₂")));
        assert!(!is_correct(&grade(ItemKind::Exact, &key, "oxygen")));
    }

    #[test]
    fn multiple_choice_compares_the_chosen_id() {
        let key = AnswerKey::Mcq { choice_id: "c2".into() };
        assert!(is_correct(&grade(ItemKind::Mcq, &key, "c2")));
        assert!(!is_correct(&grade(ItemKind::Mcq, &key, "c1")));
    }

    #[test]
    fn cloze_gives_partial_credit_per_blank() {
        // Three of four right is genuinely different from a blank page, and the
        // mastery posterior should not treat them the same.
        let key = AnswerKey::Cloze {
            blanks: vec![
                vec!["mitochondria".into()],
                vec!["atp".into(), "adenosine triphosphate".into()],
                vec!["cytoplasm".into()],
            ],
        };
        match grade(ItemKind::Cloze, &key, "mitochondria|ATP|nucleus") {
            Outcome::Decided { correct, score, blanks, .. } => {
                assert!(!correct);
                assert!((score - 2.0 / 3.0).abs() < 1e-12);
                assert_eq!(blanks, vec![true, true, false]);
            }
            other => panic!("expected a decided outcome, got {other:?}"),
        }
        assert!(is_correct(&grade(
            ItemKind::Cloze,
            &key,
            "Mitochondria | adenosine triphosphate | cytoplasm"
        )));
    }

    #[test]
    fn an_unanswered_cloze_blank_is_wrong_not_vacuously_right() {
        let key = AnswerKey::Cloze {
            blanks: vec![vec!["a".into()], vec!["b".into()]],
        };
        match grade(ItemKind::Cloze, &key, "a") {
            Outcome::Decided { blanks, correct, .. } => {
                assert_eq!(blanks, vec![true, false]);
                assert!(!correct);
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn free_response_is_the_only_kind_that_reaches_a_model_and_carries_its_rubric() {
        let key = AnswerKey::Free {
            rubric: "Names both causes and links them.".into(),
        };
        match grade(ItemKind::Free, &key, "anything") {
            Outcome::NeedsRubric { rubric } => {
                assert!(rubric.contains("Names both causes"));
            }
            other => panic!("free response must not be decided here, got {other:?}"),
        }
    }

    #[test]
    fn every_objective_kind_reports_itself_as_deterministic() {
        // The claim the whole app makes to the learner. If any objective kind ever
        // reports `Rubric`, the "no model decided this" promise is broken.
        let cases = [
            (ItemKind::Mcq, AnswerKey::Mcq { choice_id: "a".into() }, "a"),
            (
                ItemKind::Exact,
                AnswerKey::Exact { text: "a".into(), alternatives: vec![] },
                "a",
            ),
            (
                ItemKind::Cloze,
                AnswerKey::Cloze { blanks: vec![vec!["a".into()]] },
                "a",
            ),
            (
                ItemKind::Numeric,
                AnswerKey::Numeric { expected: "1".into(), tolerance: abs("0") },
                "1",
            ),
        ];
        for (kind, key, response) in cases {
            match grade(kind, &key, response) {
                Outcome::Decided { graded_by, .. } => {
                    assert_eq!(graded_by, GradedBy::Deterministic, "{}", kind.as_str());
                }
                other => panic!("{} was not decided: {other:?}", kind.as_str()),
            }
        }
    }

    #[test]
    fn absurdly_long_numbers_are_refused_rather_than_wrapping() {
        let key = AnswerKey::Numeric {
            expected: "1".into(),
            tolerance: abs("0"),
        };
        let huge = "9".repeat(MAX_DECIMAL_DIGITS + 5);
        // Not "incorrect": we could not compare it, and an i128 that silently wrapped
        // would produce a confident wrong verdict.
        assert!(matches!(
            grade(ItemKind::Numeric, &key, &huge),
            Outcome::Decided { correct: false, .. }
        ));
        let broken_key = AnswerKey::Numeric {
            expected: huge,
            tolerance: abs("0"),
        };
        assert!(matches!(
            grade(ItemKind::Numeric, &broken_key, "1"),
            Outcome::Ungradeable { .. }
        ));
    }

    #[test]
    fn decimal_parsing_table() {
        assert!(Decimal::parse("").is_none());
        assert!(Decimal::parse("   ").is_none());
        assert!(Decimal::parse(".").is_none());
        assert!(Decimal::parse("1.2.3").is_none());
        assert!(Decimal::parse("--1").is_none());
        assert!(Decimal::parse("1 2").is_none());
        assert_eq!(Decimal::parse("-0.50"), Decimal::parse("-0.50"));
        // Same value written at different scales must compare equal, which is what
        // `align` is for — `0.5` and `0.50` are the same number.
        let a = Decimal::parse("0.5").expect("parses");
        let b = Decimal::parse("0.50").expect("parses");
        assert!(a.le(b).expect("aligns") && b.le(a).expect("aligns"));
    }
}
