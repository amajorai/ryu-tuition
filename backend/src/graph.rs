//! The prerequisite DAG: what is ready to study, what to study next, and the
//! cycle check that keeps both of those questions answerable.
//!
//! An edge means "`prereq_id` must be mastered before `skill_id`". The graph is
//! small (a syllabus, not a social network) and every answer it gives is a walk over
//! it, so the only property that really matters is that it stays acyclic.
//!
//! # Where cycles are rejected, and why it is in two places
//!
//! [`crate::store::TuitionStore::add_prereq`] is the enforcement point for a single
//! edge: it walks the stored graph and inserts inside one transaction, so two
//! concurrent writes cannot each observe an acyclic graph and jointly close a cycle.
//! That check cannot cover the other writer, though. Ingest proposes a *batch* of
//! edges from one document, and a batch is only acyclic as a set — `a→b`, `b→c` and
//! `c→a` are each individually fine against a graph that does not yet contain the
//! other two. [`find_cycle`] is the pure check that runs over the whole proposed set
//! before any of it is written, which is why it lives here and not in the store.
//!
//! Rejecting at write time is not a stylistic preference. "What should I study
//! next" walks prerequisites; a cycle makes that walk non-terminating, and the only
//! alternatives to refusing the write are an app that hangs or one that silently
//! drops an edge the learner believes they added. The rejection names the offending
//! path, because "cycle detected" tells a person nothing about which edge to remove.
//!
//! # Determinism
//!
//! Adjacency is a `BTreeMap` of sorted `Vec`s and the DFS visits roots in sorted id
//! order, so the *same* cycle is reported for the same edge set on every run — a
//! test that asserts on the message is testing something stable. Same reason
//! [`study_next`] has a total tie-break chain: an answer that changes between two
//! runs over identical data is an answer a learner cannot check.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::models::{PrereqEdge, Skill, SkillStatus};

/// The posterior at which a prerequisite counts as mastered.
///
/// Mirrors [`crate::models::TuitionSettings::ready_threshold`]'s default; the
/// functions here take the threshold as an argument so the stored setting is what
/// actually applies, and this constant is the value that setting defaults to.
pub const DEFAULT_READY_THRESHOLD: f64 = 0.80;

/// A cycle in the prerequisite graph, as the path that closes it.
///
/// The path **repeats its first node at the end** (`a → b → c → a`) so the message
/// reads as a loop. Note that this differs on purpose from
/// [`crate::store`]'s internal walk, which returns an open `from → … → target` path
/// because its caller already knows the edge it was about to add closes it. Do not
/// "align" the two: the store's message and this one are each shaped for the
/// question they answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    pub path: Vec<String>,
}

impl Cycle {
    /// The path rendered through a label lookup, for a message a person can act on.
    /// Ids that resolve to nothing fall back to the id — a rejection message is not
    /// worth failing a write over.
    pub fn labelled(&self, label: impl Fn(&str) -> Option<String>) -> String {
        self.path
            .iter()
            .map(|id| label(id).unwrap_or_else(|| id.clone()))
            .collect::<Vec<_>>()
            .join(" → ")
    }
}

impl fmt::Display for Cycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.path.join(" → "))
    }
}

/// `skill → its prerequisites`, sorted and deduplicated.
///
/// Sorted because the DFS below indexes into these lists, and an unstable order
/// there would make the *reported* cycle depend on row order in the database.
fn adjacency<'a>(edges: &'a [PrereqEdge]) -> BTreeMap<&'a str, Vec<&'a str>> {
    let mut sets: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for edge in edges {
        sets.entry(edge.skill_id.as_str())
            .or_default()
            .insert(edge.prereq_id.as_str());
    }
    sets.into_iter()
        .map(|(skill, prereqs)| (skill, prereqs.into_iter().collect()))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    /// On the current DFS stack. Reaching one of these is the cycle.
    Open,
    /// Fully explored; everything below it is known acyclic.
    Done,
}

/// The first cycle in `edges`, in a stable order, or `None` if the graph is a DAG.
///
/// Iterative rather than recursive: the depth is bounded by the number of skills,
/// and a syllabus imported from a long document can carry thousands. A recursive
/// DFS would be shorter and would blow the stack on exactly the input the check
/// exists for.
pub fn find_cycle(edges: &[PrereqEdge]) -> Option<Cycle> {
    let graph = adjacency(edges);
    let mut marks: BTreeMap<&str, Mark> = BTreeMap::new();
    // Every node that has outgoing edges is a potential root. Nodes that only ever
    // appear as a prerequisite cannot start a cycle they are not also part of, and
    // they get visited as children.
    let roots: Vec<&str> = graph.keys().copied().collect();

    for root in roots {
        if marks.contains_key(root) {
            continue;
        }
        // (node, index of the next child to visit).
        let mut stack: Vec<(&str, usize)> = vec![(root, 0)];
        marks.insert(root, Mark::Open);

        while let Some(&(node, index)) = stack.last() {
            let children: &[&str] = graph.get(node).map(Vec::as_slice).unwrap_or(&[]);
            let Some(&child) = children.get(index) else {
                marks.insert(node, Mark::Done);
                stack.pop();
                continue;
            };
            stack.last_mut().expect("just peeked").1 = index + 1;
            match marks.get(child) {
                Some(Mark::Open) => {
                    // `child` is somewhere below us on the stack; the cycle is that
                    // suffix, closed by repeating the node we came back to.
                    let start = stack
                        .iter()
                        .position(|(n, _)| *n == child)
                        .expect("an open mark is on the stack");
                    let mut path: Vec<String> =
                        stack[start..].iter().map(|(n, _)| (*n).to_string()).collect();
                    path.push(child.to_string());
                    return Some(Cycle { path });
                }
                Some(Mark::Done) => {}
                None => {
                    marks.insert(child, Mark::Open);
                    stack.push((child, 0));
                }
            }
        }
    }
    None
}

/// Whether adding `skill_id → prereq_id` to `edges` would close a cycle.
///
/// The pure counterpart of the store's transactional check, for the paths that hold
/// a proposed edge set in memory (ingest review, an import) and want to reject it
/// before writing anything.
pub fn would_close_cycle(edges: &[PrereqEdge], skill_id: &str, prereq_id: &str) -> Option<Cycle> {
    if skill_id == prereq_id {
        return Some(Cycle {
            path: vec![skill_id.to_string(), skill_id.to_string()],
        });
    }
    let mut proposed = edges.to_vec();
    proposed.push(PrereqEdge {
        skill_id: skill_id.to_string(),
        prereq_id: prereq_id.to_string(),
    });
    find_cycle(&proposed)
}

/// Skills whose every prerequisite is mastered — the set that is legitimately
/// studyable right now.
///
/// Only `active` skills are candidates: a `proposed` one has not been accepted by
/// the learner and an `archived` one was deliberately put away. Prerequisites are
/// looked up across **all** statuses, though, because archiving a skill does not
/// unlearn it.
///
/// A prerequisite id that names no skill counts as **not** mastered. That is the
/// conservative direction: the graph is asserting a dependency we cannot evaluate,
/// and treating it as satisfied would hand the learner a skill whose foundation is
/// unknown. It is also unreachable through the store — `delete_skill` removes edges
/// in both directions — so seeing it means something outside the app edited the
/// database.
///
/// Returned in the input slice's order, which for
/// [`crate::store::TuitionStore::list_skills`] is weakest-first.
pub fn ready_skills<'a>(
    skills: &'a [Skill],
    edges: &[PrereqEdge],
    ready_threshold: f64,
) -> Vec<&'a Skill> {
    let mastery: BTreeMap<&str, f64> = skills
        .iter()
        .map(|skill| (skill.id.as_str(), skill.mastery))
        .collect();
    let graph = adjacency(edges);
    skills
        .iter()
        .filter(|skill| skill.status == SkillStatus::Active)
        .filter(|skill| {
            graph
                .get(skill.id.as_str())
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .all(|prereq| {
                    mastery
                        .get(prereq)
                        .is_some_and(|m| *m >= ready_threshold)
                })
        })
        .collect()
}

/// "What should I study next" — the weakest ready skill.
///
/// Ties are broken all the way down, in this order:
///
/// 1. lowest `mastery` — the point of the question;
/// 2. earliest due, where a **never-reviewed** skill uses `now`, so new work sorts
///    ahead of a review scheduled for next week but behind one that is already
///    overdue. This is the only thing `now` is used for, and it is why the function
///    takes it rather than reading a clock;
/// 3. skill id, which is total.
///
/// Without the last step two skills at identical mastery and due date would be
/// separated by nothing, and the answer would depend on row order — a learner who
/// reloads the page and is told to study something else stops trusting the page.
pub fn study_next<'a>(
    skills: &'a [Skill],
    edges: &[PrereqEdge],
    ready_threshold: f64,
    now: i64,
) -> Option<&'a Skill> {
    ready_skills(skills, edges, ready_threshold)
        .into_iter()
        .min_by(|a, b| {
            a.mastery
                .total_cmp(&b.mastery)
                .then_with(|| a.due_at.unwrap_or(now).cmp(&b.due_at.unwrap_or(now)))
                .then_with(|| a.id.cmp(&b.id))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BktParams, DEFAULT_EASE};

    fn skill(id: &str, mastery: f64, due_at: Option<i64>, status: SkillStatus) -> Skill {
        Skill {
            id: id.to_string(),
            subject_id: "sub_1".to_string(),
            name: id.to_uppercase(),
            detail: None,
            status,
            source_id: None,
            params: BktParams::default(),
            mastery,
            ease: DEFAULT_EASE,
            interval_days: 0,
            reps: 0,
            lapses: 0,
            due_at,
            last_reviewed_at: None,
            created_at: 0,
            updated_at: 0,
        }
    }

    fn edges(pairs: &[(&str, &str)]) -> Vec<PrereqEdge> {
        pairs
            .iter()
            .map(|(skill_id, prereq_id)| PrereqEdge {
                skill_id: (*skill_id).to_string(),
                prereq_id: (*prereq_id).to_string(),
            })
            .collect()
    }

    struct CycleCase {
        name: &'static str,
        edges: &'static [(&'static str, &'static str)],
        cyclic: bool,
    }

    #[test]
    fn cycles_are_found_and_dags_are_left_alone() {
        let cases = [
            CycleCase {
                name: "an empty graph is acyclic",
                edges: &[],
                cyclic: false,
            },
            CycleCase {
                name: "a chain is acyclic",
                edges: &[("c", "b"), ("b", "a")],
                cyclic: false,
            },
            CycleCase {
                name: "a diamond is acyclic — two paths to one prerequisite is fine",
                edges: &[("d", "b"), ("d", "c"), ("b", "a"), ("c", "a")],
                cyclic: false,
            },
            CycleCase {
                // THE CYCLE. Three edges, each individually harmless against a graph
                // that does not yet hold the other two — which is exactly why the
                // batch check exists alongside the store's per-edge one.
                name: "a three-edge loop is a cycle",
                edges: &[("a", "b"), ("b", "c"), ("c", "a")],
                cyclic: true,
            },
            CycleCase {
                name: "a two-edge loop is a cycle",
                edges: &[("a", "b"), ("b", "a")],
                cyclic: true,
            },
            CycleCase {
                name: "a cycle hanging off a long acyclic tail is still found",
                edges: &[("z", "y"), ("y", "x"), ("x", "m"), ("m", "n"), ("n", "m")],
                cyclic: true,
            },
        ];
        for case in cases {
            let found = find_cycle(&edges(case.edges));
            assert_eq!(found.is_some(), case.cyclic, "{}", case.name);
            if let Some(cycle) = found {
                // The reported path must actually be a loop: it closes on itself and
                // every consecutive pair is a real edge.
                assert_eq!(
                    cycle.path.first(),
                    cycle.path.last(),
                    "{}: path does not close",
                    case.name
                );
                let set: BTreeSet<(&str, &str)> = case.edges.iter().copied().collect();
                for pair in cycle.path.windows(2) {
                    assert!(
                        set.contains(&(pair[0].as_str(), pair[1].as_str())),
                        "{}: {} → {} is not an edge",
                        case.name,
                        pair[0],
                        pair[1]
                    );
                }
            }
        }
    }

    #[test]
    fn the_reported_cycle_is_the_same_one_on_every_run() {
        // Row order out of SQLite is stable but not guaranteed, and the message is
        // what a person reads to decide which edge to remove. Same edges in a
        // different order must name the same path.
        let forward = edges(&[("a", "b"), ("b", "c"), ("c", "a"), ("d", "a")]);
        let shuffled = edges(&[("d", "a"), ("c", "a"), ("b", "c"), ("a", "b")]);
        let one = find_cycle(&forward).expect("cyclic");
        let two = find_cycle(&shuffled).expect("cyclic");
        assert_eq!(one, two);
        assert_eq!(one.to_string(), "a → b → c → a");
    }

    #[test]
    fn a_cycle_is_named_with_the_skills_a_person_recognizes() {
        let cycle = find_cycle(&edges(&[("skl_a", "skl_b"), ("skl_b", "skl_a")]))
            .expect("cyclic");
        let names: BTreeMap<&str, &str> =
            [("skl_a", "Limits"), ("skl_b", "Continuity")].into_iter().collect();
        assert_eq!(
            cycle.labelled(|id| names.get(id).map(|n| (*n).to_string())),
            "Limits → Continuity → Limits"
        );
        // An id with no row falls back to the id rather than failing the message.
        assert_eq!(cycle.labelled(|_| None), "skl_a → skl_b → skl_a");
    }

    #[test]
    fn adding_an_edge_is_checked_before_it_is_written() {
        let existing = edges(&[("c", "b"), ("b", "a")]);
        // `a` requiring `c` closes a → c → b → a.
        assert!(would_close_cycle(&existing, "a", "c").is_some());
        // A new leaf does not.
        assert!(would_close_cycle(&existing, "d", "a").is_none());
        // Self-edges are their own answer, and are caught without touching the graph.
        let self_edge = would_close_cycle(&existing, "a", "a").expect("self-edge is a cycle");
        assert_eq!(self_edge.to_string(), "a → a");
    }

    #[test]
    fn a_graph_that_is_already_cyclic_terminates_instead_of_hanging() {
        // The state a `sqlite3` shell edit or an older build can leave behind. The
        // walk must return, not spin — it runs under the store's connection mutex,
        // so a hang here takes `/health` down with it.
        let mut broken = edges(&[("a", "b"), ("b", "a")]);
        broken.extend(edges(&[("c", "d"), ("d", "c")]));
        assert!(find_cycle(&broken).is_some());
        assert!(ready_skills(&[], &broken, DEFAULT_READY_THRESHOLD).is_empty());
    }

    #[test]
    fn readiness_gates_on_every_prerequisite_being_mastered() {
        let skills = vec![
            skill("skl_a", 0.95, None, SkillStatus::Active),
            skill("skl_b", 0.40, None, SkillStatus::Active),
            skill("skl_c", 0.10, None, SkillStatus::Active),
            skill("skl_d", 0.05, None, SkillStatus::Proposed),
        ];
        // c needs both a (mastered) and b (not).
        let graph = edges(&[("skl_c", "skl_a"), ("skl_c", "skl_b"), ("skl_b", "skl_a")]);
        let ready: Vec<&str> = ready_skills(&skills, &graph, DEFAULT_READY_THRESHOLD)
            .iter()
            .map(|s| s.id.as_str())
            .collect();
        // a has no prerequisites; b's only prerequisite is mastered; c is blocked by
        // b; d is only proposed and is not in the deck at all.
        assert_eq!(ready, vec!["skl_a", "skl_b"]);
    }

    #[test]
    fn a_prerequisite_that_names_nothing_blocks_rather_than_waves_through() {
        let skills = vec![skill("skl_a", 0.10, None, SkillStatus::Active)];
        let dangling = edges(&[("skl_a", "skl_gone")]);
        assert!(ready_skills(&skills, &dangling, DEFAULT_READY_THRESHOLD).is_empty());
    }

    #[test]
    fn study_next_is_the_weakest_ready_skill_and_breaks_every_tie() {
        let now = 1_786_000_000_000;
        let day = 86_400_000;
        let skills = vec![
            skill("skl_a", 0.95, Some(now - day), SkillStatus::Active),
            // Weakest overall, but blocked: its prerequisite is not mastered.
            skill("skl_blocked", 0.05, Some(now - day), SkillStatus::Active),
            skill("skl_mid", 0.40, Some(now + day), SkillStatus::Active),
            // Same mastery as skl_mid, but overdue — the due tie-break picks it.
            skill("skl_due", 0.40, Some(now - day), SkillStatus::Active),
        ];
        let graph = edges(&[("skl_blocked", "skl_mid")]);
        let next = study_next(&skills, &graph, DEFAULT_READY_THRESHOLD, now).expect("some skill");
        assert_eq!(next.id, "skl_due");

        // With mastery AND due date identical, the id decides — and the same input
        // in a different order gives the same answer.
        let twins = vec![
            skill("skl_zz", 0.40, Some(now), SkillStatus::Active),
            skill("skl_aa", 0.40, Some(now), SkillStatus::Active),
        ];
        assert_eq!(
            study_next(&twins, &[], DEFAULT_READY_THRESHOLD, now)
                .unwrap()
                .id,
            "skl_aa"
        );
        let reversed: Vec<Skill> = twins.into_iter().rev().collect();
        assert_eq!(
            study_next(&reversed, &[], DEFAULT_READY_THRESHOLD, now)
                .unwrap()
                .id,
            "skl_aa"
        );
    }

    #[test]
    fn a_never_reviewed_skill_sorts_as_due_now() {
        let now = 1_786_000_000_000;
        let day = 86_400_000;
        let skills = vec![
            // Never reviewed: new work, treated as owed as of `now`.
            skill("skl_new", 0.30, None, SkillStatus::Active),
            // Same mastery, scheduled for next week: not yet owed.
            skill("skl_later", 0.30, Some(now + 7 * day), SkillStatus::Active),
        ];
        assert_eq!(
            study_next(&skills, &[], DEFAULT_READY_THRESHOLD, now)
                .unwrap()
                .id,
            "skl_new"
        );
        // …but an already-overdue review still comes first.
        let mut with_overdue = skills;
        with_overdue.push(skill("skl_overdue", 0.30, Some(now - day), SkillStatus::Active));
        assert_eq!(
            study_next(&with_overdue, &[], DEFAULT_READY_THRESHOLD, now)
                .unwrap()
                .id,
            "skl_overdue"
        );
    }

    #[test]
    fn nothing_ready_is_none_not_a_panic() {
        let skills = vec![skill("skl_a", 0.10, None, SkillStatus::Archived)];
        assert!(study_next(&skills, &[], DEFAULT_READY_THRESHOLD, 0).is_none());
        assert!(study_next(&[], &[], DEFAULT_READY_THRESHOLD, 0).is_none());
    }
}
