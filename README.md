<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./icon-dark.png" />
    <img src="./icon-light.png" alt="Tuition" width="144" />
  </picture>
</p>

<div align="center">

# Tuition

</div>

A tutor for one learner: turn your own syllabus and notes into a prerequisite graph of skills, each carrying a Bayesian Knowledge Tracing posterior, and drill the weakest thing you are ready for. Four of the five question kinds are graded by arithmetic with no model involved, on SM-2 review scheduling.

> **The public home of `ryu-tuition`.** Source, builds, and releases live here —
> binaries for every platform are attached to each release.
>
> This tree is generated from the Ryu monorepo, so commits pushed here
> directly are replaced on the next sync. **Pull requests are welcome** —
> open them here and they are ported into the monorepo, then flow back out.
> Ryu as a whole: https://github.com/amajorai/ryu

## Install

**App:** [Install](ryu://apps/@ryu/tuition) (opens the Ryu desktop app and asks you to confirm)

**CLI:**

```bash
ryu apps add @ryu/tuition
```

**Crate:**

```bash
cargo install ryu-tuition
```

Prebuilt binaries for every platform are attached to [each release](https://github.com/amajorai/ryu/releases).

## License

Apache-2.0 — see [LICENSE](./LICENSE).

## What it does, concretely

- **Ingest.** Drop in a syllabus, a chapter, a set of notes. Rich formats go through
  whatever `document.parse` provider is installed on the node (Docling, MinerU, MarkItDown,
  Unstructured — bound by capability, never hardcoded). A model then proposes skills and
  prerequisite edges, **and you review them before they take effect.**
- **Practice.** Items are generated per skill from your own source material, stored,
  versioned and reused. Multiple-choice, cloze, numeric with an explicit tolerance,
  exact-match, and free response.
- **Grading you can check.** Every objective item kind is graded by comparison, with **no
  model in the loop at all** — a numeric answer is decided by decimal arithmetic against a
  stated tolerance, not by a model's opinion of whether you were close enough. Free
  response is the only kind a model marks, and it is marked against a written rubric that
  is shown with the grade.
- **Mastery, not streaks.** Each attempt updates a Bayesian Knowledge Tracing posterior per
  skill. The number on screen is `P(you know this)`, and the app will tell you how many more
  correct attempts get you to your target. Multiple-choice questions use a guess rate
  derived from the number of choices, because scoring a two-choice question with a
  four-choice guess rate is how a mastery model quietly lies to you.
- **Review scheduling.** SM-2 over *skills* rather than cards, on whole days from a day
  boundary in your timezone — so a review due "tomorrow" does not shift because you studied
  at 23:58.
- **What to study next.** Prerequisites form a DAG (cycles are rejected when you save them,
  with the offending path named). "Next" is the lowest-mastery skill whose prerequisites you
  have already met — a topological answer, stable across runs.
- **Sessions that fit.** Tell it you have 25 minutes; it picks the set of due items with the
  best expected mastery gain per minute, capped so one weak skill cannot eat the whole
  session.
- **Exam trajectory.** Given an exam date and your observed rate of progress, it projects
  whether you will cover the syllabus in time — and says `unknown` rather than
  extrapolating when it has fewer than three sessions of history.

## How it uses the rest of Ryu

- **Your chats become your revision deck.** Turn on *Study mode* in the composer and a
  `post_assistant_turn` hook takes what an answer just explained and queues it as
  *candidate* review items for the active subject. Candidates are queued, never
  auto-accepted — you approve what enters the deck. This is the human mirror of what
  `@ryu/learning` does for agent skills.
- **Any agent can tutor you.** The app ships an MCP server: `tuition__due`,
  `tuition__quiz`, `tuition__grade`, `tuition__log` and `tuition__mastery`. A chat, a
  subagent or a workflow `mcp` node can serve you questions and record the attempts into the
  same model the companion reads.
- **Events worth routing.** `review.due`, `mastery.dropped` and `goal.at-risk` are declared
  `hook_events`, so notifications, the approval Inbox and workflows can all bind to them.
- **Document parsing** is a capability request, not a dependency on a particular parser.

## Architecture

`apps-store/tuition/` is a self-contained satellite. `backend/` is the `ryu-tuition`
sidecar binary (axum + rusqlite, no lib target, no dependency on `apps/core`), reached
through the generic ext-proxy at `/api/tuition/*`. `ui/` is the sandboxed companion
(`vite-plugin-singlefile`, CSP `connect-src 'none'`), which talks to its own sidecar
through one generic forwarder rather than a verb per endpoint. The only thing that lands in
Core is the registration.

## Privacy

Everything (your materials, your items, every attempt) stays in the node's own SQLite
database. Model calls happen at exactly two edges (proposing skills from a document, and
generating or rubric-grading items) and both are explicit. The scheduling and mastery math
never leaves the machine and never calls anything.
