//! The session planner: fill the minutes the learner actually has with the items
//! that move the mastery model the most.
//!
//! A knapsack, solved greedily by `gain / cost`. Not optimally, and deliberately
//! not: the exact solution is pseudo-polynomial in the budget, the cost estimates
//! feeding it are medians over a handful of observations, and an optimum computed
//! from noisy inputs is precision theatre. Greedy by ratio is what a person would do
//! with the same information, and it is explainable line by line — every planned
//! item carries the cost and gain that put it there
//! ([`crate::models::SessionItem::est_cost_ms`] / `est_gain`), so a finished session
//! can be compared against what the planner predicted.
//!
//! # The two weightings that are not obvious
//!
//! **Gain is weighted by `1 - m`.** The raw BKT gain `m' - m` peaks in the middle of
//! the curve, but a skill at 0.88 heading for 0.90 is *nearly done* and does not
//! deserve the same claim on a 20-minute sitting as one at 0.30. Multiplying by the
//! remaining distance to certainty is what stops a nearly-mastered skill from
//! hogging the session it no longer needs.
//!
//! **Cost is a median, not a mean.** One item the learner walked away from
//! mid-answer contributes a 40-minute latency, and a mean over five observations
//! would then price that whole item kind out of every future session.

use std::collections::BTreeMap;

use crate::bkt;
use crate::models::{BktParams, ItemKind, PlannedItem};

/// The planner's seed cost for an item kind it has never seen answered, in seconds.
/// Mirrors [`crate::models::TuitionSettings::default_item_seconds`].
pub const DEFAULT_ITEM_SECONDS: u32 = 60;

/// Items from one skill in a single sitting. Mirrors
/// [`crate::models::TuitionSettings::per_skill_item_cap`].
///
/// Without it the greedy pass fills the whole session from the single weakest
/// skill — which is what "maximize expected gain" literally asks for, and which
/// produces a twenty-minute sitting on one topic that a learner abandons.
pub const DEFAULT_PER_SKILL_ITEM_CAP: u32 = 5;

/// Floor on an item's estimated cost, in milliseconds.
///
/// The ratio is `gain / cost`, so a zero (or negative, from a corrupt latency row)
/// cost is either a division by zero or an item that sorts above everything while
/// claiming to be free. One second is below any real answer and keeps the ordering
/// meaningful.
pub const MIN_ITEM_COST_MS: i64 = 1_000;

/// One item the planner may choose, with everything it needs to price and value it.
///
/// Borrows its ids: the caller holds the [`crate::models::Item`] rows this was built
/// from, and the plan that comes back is copied into the session anyway.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate<'a> {
    pub item_id: &'a str,
    pub skill_id: &'a str,
    pub kind: ItemKind,
    /// Choices on the item, for the MCQ guess-rate floor. Zero for other kinds.
    pub choice_count: usize,
    /// The skill's current posterior.
    pub mastery: f64,
    pub params: BktParams,
}

/// Observed answer times per item kind, with a seed for the kinds that have none.
///
/// Built from [`crate::store::TuitionStore::latencies_for_kind`], which returns
/// observations **sorted by latency ascending** — so pass it a limit at or above the
/// number of attempts that exist, or the median you take is the median of the
/// fastest N answers rather than of the distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct CostModel {
    fallback_ms: i64,
    /// A short association list rather than a map: there are five item kinds and
    /// [`ItemKind`] is not `Ord`.
    medians: Vec<(ItemKind, i64)>,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::seeded(DEFAULT_ITEM_SECONDS)
    }
}

impl CostModel {
    /// A model with no observations at all: every kind costs the seed.
    pub fn seeded(default_seconds: u32) -> Self {
        Self {
            fallback_ms: (i64::from(default_seconds) * 1_000).max(MIN_ITEM_COST_MS),
            medians: Vec::new(),
        }
    }

    /// Record the observed median for one kind. A `None` (or empty) observation set
    /// leaves the kind on the seed.
    pub fn observe(mut self, kind: ItemKind, ascending_latencies: &[i64]) -> Self {
        if let Some(median) = median_ms(ascending_latencies) {
            self.medians.push((kind, median.max(MIN_ITEM_COST_MS)));
        }
        self
    }

    pub fn cost_ms(&self, kind: ItemKind) -> i64 {
        self.medians
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, ms)| *ms)
            .unwrap_or(self.fallback_ms)
    }
}

/// Median of an **ascending** slice. `None` for an empty one.
///
/// The even case averages the two central values, which for latencies in
/// milliseconds is exact enough that rounding direction never changes an ordering —
/// but it is fixed (truncating toward zero) rather than left to a float, because the
/// planner's output has to be reproducible.
pub fn median_ms(ascending: &[i64]) -> Option<i64> {
    match ascending.len() {
        0 => None,
        n if n % 2 == 1 => Some(ascending[n / 2]),
        n => {
            let lo = ascending[n / 2 - 1];
            let hi = ascending[n / 2];
            Some(lo / 2 + hi / 2 + (lo % 2 + hi % 2) / 2)
        }
    }
}

/// What the planner was asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanOptions {
    pub budget_minutes: u32,
    pub per_skill_item_cap: u32,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            budget_minutes: 20,
            per_skill_item_cap: DEFAULT_PER_SKILL_ITEM_CAP,
        }
    }
}

/// The expected gain of asking one item, weighted so nearly-mastered skills stop
/// competing for the session.
///
/// Uses the same floored guess rate the grader will apply, so the number on the plan
/// is the number the attempt actually produces.
pub fn expected_gain(candidate: &Candidate<'_>) -> f64 {
    let update = bkt::update_for_item(
        candidate.mastery,
        candidate.params,
        candidate.kind,
        candidate.choice_count,
        true,
    );
    let raw = (update.after - update.before).max(0.0);
    raw * (1.0 - update.before)
}

/// Greedy knapsack: pick items by descending `gain / cost` while they fit.
///
/// Two decisions the loop makes that a reader might undo:
///
/// - It **keeps scanning after an item does not fit** rather than stopping. A
///   candidate that overruns the remaining budget does not preclude a cheaper one
///   further down, and stopping at the first miss leaves the last few minutes of
///   every session unspent.
/// - Ties are broken by `item_id`, so two items with identical gain and cost always
///   plan in the same order. Without it the plan depends on the row order the store
///   happened to return, and "resume the sitting you started" would resume a
///   different one.
pub fn plan<'a>(
    candidates: &[Candidate<'a>],
    costs: &CostModel,
    options: PlanOptions,
) -> Vec<PlannedItem> {
    let budget_ms = i64::from(options.budget_minutes) * 60_000;
    if budget_ms <= 0 || candidates.is_empty() {
        return Vec::new();
    }

    let mut ranked: Vec<(f64, i64, &Candidate<'a>)> = candidates
        .iter()
        .map(|candidate| {
            let cost = costs.cost_ms(candidate.kind).max(MIN_ITEM_COST_MS);
            let gain = expected_gain(candidate);
            let ratio = gain / cost as f64;
            (ratio, cost, candidate)
        })
        .collect();
    // `total_cmp` rather than `partial_cmp().unwrap()`: the ratio is a division and
    // an unwrap here would be a panic in the one code path a learner starts a
    // session through.
    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.2.item_id.cmp(b.2.item_id))
    });

    let mut spent = 0i64;
    let mut per_skill: BTreeMap<&str, u32> = BTreeMap::new();
    let mut plan = Vec::new();
    for (_, cost, candidate) in ranked {
        if spent.saturating_add(cost) > budget_ms {
            continue;
        }
        let taken = per_skill.entry(candidate.skill_id).or_insert(0);
        if *taken >= options.per_skill_item_cap {
            continue;
        }
        *taken += 1;
        spent += cost;
        plan.push(PlannedItem {
            item_id: candidate.item_id.to_string(),
            skill_id: candidate.skill_id.to_string(),
            est_cost_ms: cost,
            est_gain: expected_gain(candidate),
        });
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate<'a>(item_id: &'a str, skill_id: &'a str, mastery: f64) -> Candidate<'a> {
        Candidate {
            item_id,
            skill_id,
            kind: ItemKind::Exact,
            choice_count: 0,
            mastery,
            params: BktParams::default(),
        }
    }

    #[test]
    fn the_median_cost_is_the_middle_observation_not_the_mean() {
        // One abandoned answer must not price a whole item kind out of the session.
        let with_an_outlier = [8_000, 9_000, 10_000, 11_000, 2_400_000];
        assert_eq!(median_ms(&with_an_outlier), Some(10_000));
        assert_eq!(median_ms(&[10_000, 20_000]), Some(15_000));
        assert_eq!(median_ms(&[1, 2]), Some(1));
        assert_eq!(median_ms(&[]), None);
        // An empty observation set leaves the kind on its seed.
        let costs = CostModel::seeded(60).observe(ItemKind::Mcq, &[]);
        assert_eq!(costs.cost_ms(ItemKind::Mcq), 60_000);
    }

    #[test]
    fn a_nearly_mastered_skill_does_not_outrank_a_weak_one() {
        // The `1 - m` weighting, stated as an ordering. Raw BKT gain peaks mid-curve,
        // so without the weighting the 0.85 skill can outrank the 0.10 one.
        let weak = expected_gain(&candidate("itm_weak", "skl_weak", 0.10));
        let middling = expected_gain(&candidate("itm_mid", "skl_mid", 0.50));
        let nearly = expected_gain(&candidate("itm_done", "skl_done", 0.95));
        assert!(nearly < middling);
        assert!(nearly < weak);
    }

    #[test]
    fn the_plan_fits_the_budget_and_is_ordered_by_value_per_minute() {
        let costs = CostModel::seeded(60); // every item costs 60s
        let candidates = vec![
            candidate("itm_a", "skl_a", 0.10),
            candidate("itm_b", "skl_b", 0.40),
            candidate("itm_c", "skl_c", 0.95),
        ];
        let plan = plan(
            &candidates,
            &costs,
            PlanOptions {
                budget_minutes: 2,
                per_skill_item_cap: DEFAULT_PER_SKILL_ITEM_CAP,
            },
        );
        // Two minutes buys two items, and they are the two most valuable ones.
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].item_id, "itm_a");
        assert_eq!(plan[1].item_id, "itm_b");
        let total: i64 = plan.iter().map(|p| p.est_cost_ms).sum();
        assert!(total <= 2 * 60_000);
        // The estimates travel with the plan, or a finished session cannot be
        // compared against what was predicted.
        assert!(plan[0].est_gain > plan[1].est_gain);
        assert_eq!(plan[0].est_cost_ms, 60_000);
    }

    #[test]
    fn an_item_that_does_not_fit_does_not_end_the_plan() {
        // The case that fails if the loop `break`s instead of continuing: the most
        // valuable item is also the most expensive one and overruns the budget, but
        // a cheap item further down still fits.
        let costs = CostModel::seeded(60)
            .observe(ItemKind::Free, &[600_000]) // 10 minutes
            .observe(ItemKind::Mcq, &[30_000]); // 30 seconds
        let expensive = Candidate {
            kind: ItemKind::Free,
            ..candidate("itm_expensive", "skl_a", 0.05)
        };
        let cheap = Candidate {
            kind: ItemKind::Mcq,
            choice_count: 4,
            ..candidate("itm_cheap", "skl_b", 0.30)
        };
        let plan = plan(
            &[expensive, cheap],
            &costs,
            PlanOptions {
                budget_minutes: 1,
                per_skill_item_cap: DEFAULT_PER_SKILL_ITEM_CAP,
            },
        );
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].item_id, "itm_cheap");
    }

    #[test]
    fn one_weak_skill_cannot_consume_the_whole_session() {
        let costs = CostModel::seeded(60);
        // Ten items on the weakest skill, one on a stronger one. Uncapped, the weak
        // skill would take every slot the budget allows.
        let mut candidates: Vec<Candidate> = (0..10)
            .map(|i| {
                let id: &'static str = Box::leak(format!("itm_weak_{i}").into_boxed_str());
                candidate(id, "skl_weak", 0.10)
            })
            .collect();
        candidates.push(candidate("itm_other", "skl_other", 0.35));
        let plan = plan(
            &candidates,
            &costs,
            PlanOptions {
                budget_minutes: 10,
                per_skill_item_cap: 3,
            },
        );
        let from_weak = plan.iter().filter(|p| p.skill_id == "skl_weak").count();
        assert_eq!(from_weak, 3);
        assert!(plan.iter().any(|p| p.skill_id == "skl_other"));
    }

    #[test]
    fn the_same_candidates_always_plan_in_the_same_order() {
        let costs = CostModel::seeded(60);
        // Identical gain and cost: only the id tie-break separates them.
        let forward = vec![
            candidate("itm_zz", "skl_a", 0.40),
            candidate("itm_aa", "skl_b", 0.40),
        ];
        let reversed: Vec<Candidate> = forward.iter().copied().rev().collect();
        let one = plan(&forward, &costs, PlanOptions::default());
        let two = plan(&reversed, &costs, PlanOptions::default());
        assert_eq!(one, two);
        assert_eq!(one[0].item_id, "itm_aa");
    }

    #[test]
    fn a_zero_budget_plans_nothing_and_nothing_panics() {
        let costs = CostModel::seeded(60);
        let candidates = vec![candidate("itm_a", "skl_a", 0.10)];
        assert!(plan(
            &candidates,
            &costs,
            PlanOptions {
                budget_minutes: 0,
                per_skill_item_cap: 5
            }
        )
        .is_empty());
        assert!(plan(&[], &costs, PlanOptions::default()).is_empty());
        // A corrupt latency row cannot make an item free — and so cannot make its
        // ratio infinite and take the whole session.
        let broken = CostModel::seeded(60).observe(ItemKind::Exact, &[0, 0, 0]);
        assert_eq!(broken.cost_ms(ItemKind::Exact), MIN_ITEM_COST_MS);
    }
}
