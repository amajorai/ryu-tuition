//! Tuition: a tutor for one learner, whose every consequential number is computed
//! rather than generated.
//!
//! ```text
//!   syllabus / chapter / notes ──(document.parse)──▶ text ──(model)──▶ proposed
//!                                                                     skills + edges
//!                                                                          │
//!                                                            a person accepts them
//!                                                                          ▼
//!                     ┌───────────────────────────────────────────────────────┐
//!                     │  DETERMINISTIC SPINE — no model anywhere inside        │
//!                     │  BKT posterior · SM-2 schedule · prerequisite DAG ·    │
//!                     │  session planner · exam trajectory · objective grading │
//!                     └───────────────────────────────────────────────────────┘
//!                                                                          ▲
//!   free-response answer ──(model, against the item's written rubric)───────┘
//! ```
//!
//! A model is used at exactly two edges — proposing structure from a document, and
//! writing items / marking free responses — and **nowhere in the middle**. Whether
//! a multiple-choice answer was right, when a skill is next owed, which skill is
//! ready to study, whether the exam date is reachable: all arithmetic over stored
//! state, reproducible offline, checkable by hand. That is the whole design. A
//! mastery number a learner cannot audit is a number they eventually stop believing,
//! and once they stop believing it the schedule built on it is worthless.
//!
//! # Layout
//!
//! | module | role |
//! |--------|------|
//! | [`paths`] | data-dir resolution, so the sidecar opens the node's own `tuition.db` |
//! | [`models`] | the wire + domain types, the id/time helpers, the settings defaults |
//! | [`error`] | the one `ApiError` every handler returns |
//! | [`state`] | `AppState`, `Config::from_env`, the plugin id and its event ids |
//! | [`store`] | every SQL statement in the crate, one method per operation |
//! | [`bkt`] | Bayesian Knowledge Tracing — the mastery posterior and its projection |
//! | [`srs`] | SM-2 review scheduling over skills, on learner-local day boundaries |
//! | [`graph`] | the prerequisite DAG: cycle rejection, ready set, "study next" |
//! | [`planner`] | filling a minutes budget by expected mastery gain per minute |
//! | [`trajectory`] | the exam-date projection, and when to refuse to make one |
//! | [`grade`] | objective grading, with a hand-rolled decimal and no model at all |
//!
//! # Why there is a `[lib]` at all
//!
//! This crate is a process, not a library Core links — Core reaches it exclusively
//! through the generic ext-proxy, and nothing outside this crate depends on it. The
//! lib target exists so the deterministic spine is unit-testable on its own, and so
//! the `mcp` subcommand and the HTTP surface share one implementation rather than
//! two that drift. `ryu-reasoning` has the same shape for the same reason.

pub mod api;
pub mod error;
pub mod host;
pub mod mcp;
pub mod models;
pub mod paths;
pub mod service;
pub mod state;
pub mod store;
pub mod tick;

// The deterministic spine. Every one of these is a pure function of stored state
// with `now` passed IN rather than read inside, so a verdict is reproducible: the
// same attempt history replays to the same mastery, the same due dates and the same
// session plan, on any machine, offline, forever.
pub mod bkt;
pub mod grade;
pub mod graph;
pub mod planner;
pub mod srs;
pub mod trajectory;
