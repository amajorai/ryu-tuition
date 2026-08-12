//! The HTTP surface, mounted at `/api/tuition`.
//!
//! Handlers are thin on purpose: anything that decides something lives in
//! [`crate::service`] or the engine modules, so the companion and the MCP server
//! cannot drift apart. What is left here is extraction, the JSON envelope, and turning
//! a missing row into a 404.
//!
//! # Every route here must also be in the manifest
//!
//! Core's ext-proxy matches the declared `http.routes[]` with an EXACT segment count,
//! so a route this router serves but the manifest does not declare is a hard 404 that
//! reads like a bug in this file. The two lists are checked against each other by
//! `every_served_route_is_declared_in_the_manifest` at the bottom.
//!
//! # Envelope
//!
//! Lists come back as `{"<plural>": [...]}`, single entities at the top level,
//! mutations that return nothing as `{"ok": true}`. Creates return 200 with the
//! entity, never 201, and nothing ever returns 204 — matching `@ryu/social`, so a
//! client written against one app's shape works against the other.

use axum::{
    extract::{Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    error::{ApiError, ApiResult},
    models::{
        now_ms, AnswerKey, CandidateStatus, Choice, ItemKind, ItemOrigin, NewItem, SkillStatus,
        SourceKind, TuitionSettings,
    },
    service,
    state::AppState,
};

/// Default list size, and the ceiling a caller can ask for.
const DEFAULT_LIMIT: i64 = 200;
const MAX_LIMIT: i64 = 500;

/// Every path this router serves, relative to the mount.
///
/// Kept as data so the manifest cross-check can read it. It is duplicated from the
/// `.route()` calls below rather than derived, because axum's `Router` does not expose
/// its table — so the test asserts BOTH directions against the manifest and this list
/// is what makes a missed route visible.
pub const SERVED_ROUTES: &[&str] = &[
    "/subjects",
    "/subjects/:id",
    "/sources",
    "/sources/:id",
    "/skills",
    "/skills/:id",
    "/skills/:id/prereqs",
    "/items",
    "/items/generate",
    "/items/:id",
    "/sessions",
    "/sessions/plan",
    "/sessions/:id",
    "/sessions/:id/next",
    "/sessions/:id/answer",
    "/sessions/:id/finish",
    "/attempts",
    "/mastery",
    "/due",
    "/trajectory",
    "/candidates",
    "/candidates/:id/accept",
    "/candidates/:id/reject",
    "/settings",
];

pub fn routes(state: AppState) -> Router<()> {
    Router::new()
        .route("/subjects", get(list_subjects).post(create_subject))
        .route(
            "/subjects/:id",
            get(get_subject).patch(update_subject).delete(delete_subject),
        )
        .route("/sources", get(list_sources).post(create_source))
        .route("/sources/:id", get(get_source).delete(delete_source))
        .route("/skills", get(list_skills).post(create_skill))
        .route(
            "/skills/:id",
            get(get_skill).patch(update_skill).delete(delete_skill),
        )
        .route(
            "/skills/:id/prereqs",
            get(list_prereqs).post(add_prereq).delete(remove_prereq),
        )
        .route("/items", get(list_items).post(create_item))
        .route("/items/generate", post(generate_items))
        .route("/items/:id", get(get_item).patch(update_item).delete(delete_item))
        .route("/sessions", get(list_sessions))
        .route("/sessions/plan", post(plan_session))
        .route("/sessions/:id", get(get_session))
        .route("/sessions/:id/next", get(next_item))
        .route("/sessions/:id/answer", post(answer))
        .route("/sessions/:id/finish", post(finish_session))
        .route("/attempts", get(list_attempts))
        .route("/mastery", get(mastery))
        .route("/due", get(due))
        .route("/trajectory", get(trajectory))
        .route("/candidates", get(list_candidates))
        .route("/candidates/:id/accept", post(accept_candidate))
        .route("/candidates/:id/reject", post(reject_candidate))
        .route("/settings", get(get_settings).put(put_settings))
        .with_state(state)
}

// ── Query shapes ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SubjectQuery {
    pub subject_id: Option<String>,
    pub limit: Option<i64>,
}

impl SubjectQuery {
    fn limit(&self) -> i64 {
        self.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT)
    }

    fn subject(&self) -> ApiResult<&str> {
        self.subject_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::BadRequest("subject_id is required".into()))
    }
}

#[derive(Debug, Deserialize)]
pub struct SkillQuery {
    pub skill_id: Option<String>,
    pub include_archived: Option<bool>,
    pub limit: Option<i64>,
}

/// Turn a store's "did anything change" into a 404.
fn require_hit(changed: bool, what: &str) -> ApiResult<Json<Value>> {
    if changed {
        Ok(Json(json!({ "ok": true })))
    } else {
        Err(ApiError::NotFound(what.to_owned()))
    }
}

// ── Subjects ───────────────────────────────────────────────────────────────────

async fn list_subjects(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let subjects = state.store.list_subjects().await?;
    Ok(Json(json!({ "subjects": subjects })))
}

#[derive(Debug, Deserialize)]
struct SubjectBody {
    name: String,
    detail: Option<String>,
    exam_date: Option<String>,
    timezone: Option<String>,
}

async fn create_subject(
    State(state): State<AppState>,
    Json(body): Json<SubjectBody>,
) -> ApiResult<Json<Value>> {
    if body.name.trim().is_empty() {
        return Err(ApiError::BadRequest("a subject needs a name".into()));
    }
    let subject = state
        .store
        .create_subject(
            &body.name,
            body.detail.as_deref(),
            body.exam_date.as_deref(),
            body.timezone.as_deref(),
        )
        .await?;
    Ok(Json(serde_json::to_value(subject)?))
}

async fn get_subject(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let subject = state
        .store
        .get_subject(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("subject".into()))?;
    Ok(Json(serde_json::to_value(subject)?))
}

async fn update_subject(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SubjectBody>,
) -> ApiResult<Json<Value>> {
    let changed = state
        .store
        .update_subject(
            &id,
            &body.name,
            body.detail.as_deref(),
            body.exam_date.as_deref(),
            body.timezone.as_deref().unwrap_or("UTC"),
        )
        .await?;
    require_hit(changed, "subject")
}

async fn delete_subject(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_subject(&id).await?, "subject")
}

// ── Sources ────────────────────────────────────────────────────────────────────

async fn list_sources(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let sources = state.store.list_sources(query.subject()?).await?;
    Ok(Json(json!({ "sources": sources })))
}

#[derive(Debug, Deserialize)]
struct SourceBody {
    subject_id: String,
    title: String,
    kind: Option<String>,
    uri: Option<String>,
    /// The already-extracted text. Rich formats are converted before they reach this
    /// app; nothing here parses a PDF.
    text: String,
    parser: Option<String>,
}

async fn create_source(
    State(state): State<AppState>,
    Json(body): Json<SourceBody>,
) -> ApiResult<Json<Value>> {
    if body.text.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "a source needs some text to read".into(),
        ));
    }
    let source = state
        .store
        .create_source(
            &body.subject_id,
            SourceKind::parse(body.kind.as_deref().unwrap_or("")),
            &body.title,
            body.uri.as_deref(),
            &body.text,
            body.parser.as_deref(),
        )
        .await?;
    Ok(Json(serde_json::to_value(source)?))
}

async fn get_source(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let source = state
        .store
        .get_source(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("source".into()))?;
    Ok(Json(serde_json::to_value(source)?))
}

async fn delete_source(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_source(&id).await?, "source")
}

// ── Skills ─────────────────────────────────────────────────────────────────────

async fn list_skills(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let skills = state.store.list_skills(query.subject()?, None).await?;
    let edges = state.store.list_prereq_edges(query.subject()?).await?;
    Ok(Json(json!({ "skills": skills, "prereqs": edges })))
}

#[derive(Debug, Deserialize)]
struct SkillBody {
    subject_id: Option<String>,
    name: String,
    detail: Option<String>,
    status: Option<String>,
}

async fn create_skill(
    State(state): State<AppState>,
    Json(body): Json<SkillBody>,
) -> ApiResult<Json<Value>> {
    let subject_id = body
        .subject_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("subject_id is required".into()))?;
    // Default ACTIVE, not proposed. `SkillStatus::parse` falls back to `Proposed`,
    // which is right for the ingest path — a model proposes skills and a human
    // reviews them, and that review step is the point — but wrong here: a learner
    // typing a skill into the companion has already accepted it. Defaulting to
    // proposed made a hand-created skill invisible to scheduling forever, so it was
    // never planned, never attempted and never due. The ingest path passes
    // `"proposed"` explicitly.
    let status = match body.status.as_deref() {
        Some(raw) if !raw.trim().is_empty() => SkillStatus::parse(raw),
        _ => SkillStatus::Active,
    };
    let skill = state
        .store
        .upsert_skill(
            subject_id,
            &body.name,
            body.detail.as_deref(),
            status,
            None,
            Default::default(),
        )
        .await?;
    Ok(Json(serde_json::to_value(skill)?))
}

async fn get_skill(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let skill = state
        .store
        .get_skill(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("skill".into()))?;
    let prereqs = state.store.list_prereqs(&id).await?;
    let mut value = serde_json::to_value(skill)?;
    value["prereqs"] = json!(prereqs);
    Ok(Json(value))
}

async fn update_skill(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SkillBody>,
) -> ApiResult<Json<Value>> {
    let existing = state
        .store
        .get_skill(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("skill".into()))?;
    let changed = state
        .store
        .update_skill(
            &id,
            &body.name,
            body.detail.as_deref(),
            SkillStatus::parse(body.status.as_deref().unwrap_or("")),
            existing.params,
        )
        .await?;
    require_hit(changed, "skill")
}

async fn delete_skill(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_skill(&id).await?, "skill")
}

async fn list_prereqs(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    Ok(Json(json!({ "prereqs": state.store.list_prereqs(&id).await? })))
}

#[derive(Debug, Deserialize)]
struct PrereqBody {
    prereq_id: String,
}

async fn add_prereq(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<PrereqBody>,
) -> ApiResult<Json<Value>> {
    // The store rejects a cycle and names the path; surfacing it as a 400 rather than
    // a 500 is the difference between "you cannot do that, here is why" and "the app
    // broke".
    state
        .store
        .add_prereq(&id, &body.prereq_id)
        .await
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    Ok(Json(json!({ "ok": true })))
}

async fn remove_prereq(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<PrereqBody>,
) -> ApiResult<Json<Value>> {
    require_hit(
        state.store.remove_prereq(&id, &body.prereq_id).await?,
        "prerequisite",
    )
}

// ── Items ──────────────────────────────────────────────────────────────────────

async fn list_items(
    State(state): State<AppState>,
    Query(query): Query<SkillQuery>,
) -> ApiResult<Json<Value>> {
    let limit = query.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let include_archived = query.include_archived.unwrap_or(false);
    let skill_id = query
        .skill_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest("skill_id is required".into()))?;
    let items = state
        .store
        .list_items(skill_id, include_archived, limit)
        .await?;
    Ok(Json(json!({ "items": items })))
}

#[derive(Debug, Deserialize)]
struct ItemBody {
    skill_id: String,
    kind: String,
    prompt: String,
    #[serde(default)]
    choices: Vec<Choice>,
    answer: AnswerKey,
    source_id: Option<String>,
    source_ref: Option<String>,
}

async fn create_item(
    State(state): State<AppState>,
    Json(body): Json<ItemBody>,
) -> ApiResult<Json<Value>> {
    let item = state
        .store
        .create_item(&NewItem {
            skill_id: body.skill_id,
            kind: ItemKind::parse(&body.kind),
            prompt: body.prompt,
            choices: body.choices,
            answer: body.answer,
            origin: ItemOrigin::Human,
            origin_model: None,
            source_id: body.source_id,
            source_ref: body.source_ref,
        })
        .await?;
    Ok(Json(serde_json::to_value(item)?))
}

async fn get_item(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let item = state
        .store
        .get_item(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("item".into()))?;
    Ok(Json(serde_json::to_value(item)?))
}

#[derive(Debug, Deserialize)]
struct ItemPatch {
    prompt: String,
    #[serde(default)]
    choices: Vec<Choice>,
    answer: AnswerKey,
    source_ref: Option<String>,
}

async fn update_item(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ItemPatch>,
) -> ApiResult<Json<Value>> {
    let changed = state
        .store
        .update_item(
            &id,
            &body.prompt,
            &body.choices,
            &body.answer,
            body.source_ref.as_deref(),
        )
        .await?;
    require_hit(changed, "item")
}

async fn delete_item(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_item(&id).await?, "item")
}

#[derive(Debug, Deserialize)]
struct GenerateBody {
    skill_id: String,
    count: Option<u32>,
}

async fn generate_items(
    State(state): State<AppState>,
    Json(body): Json<GenerateBody>,
) -> ApiResult<Json<Value>> {
    let Some(host) = state.host.as_ref() else {
        return Err(ApiError::Unavailable(crate::host::no_host_message().into()));
    };
    let count = body.count.unwrap_or(5).clamp(1, 20);
    let items = service::generate_items(&state, host, &body.skill_id, count).await?;
    Ok(Json(json!({ "items": items })))
}

// ── Sessions ───────────────────────────────────────────────────────────────────

async fn list_sessions(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let sessions = state
        .store
        .list_sessions(query.subject()?, query.limit())
        .await?;
    Ok(Json(json!({ "sessions": sessions })))
}

#[derive(Debug, Deserialize)]
struct PlanBody {
    subject_id: String,
    minutes: Option<u32>,
}

async fn plan_session(
    State(state): State<AppState>,
    Json(body): Json<PlanBody>,
) -> ApiResult<Json<Value>> {
    let now = now_ms();
    let minutes = body.minutes.unwrap_or(15).clamp(1, 240);
    let plan = service::plan_session(&state, &body.subject_id, minutes, now).await?;
    if plan.is_empty() {
        // Not an error: nothing is due. Saying so plainly beats an empty session the
        // learner then has to work out the meaning of.
        return Ok(Json(json!({
            "session": Value::Null,
            "planned": [],
            "note": "nothing is due right now"
        })));
    }
    let session = state
        .store
        .create_session(&body.subject_id, minutes, &plan)
        .await?;
    state.store.start_session(&session.id, now).await?;
    Ok(Json(json!({ "session": session, "planned": plan })))
}

async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let session = state
        .store
        .get_session(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("session".into()))?;
    let items = state.store.list_session_items(&id).await?;
    Ok(Json(json!({ "session": session, "items": items })))
}

async fn next_item(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let Some(entry) = state.store.next_session_item(&id).await? else {
        return Ok(Json(json!({ "item": Value::Null, "done": true })));
    };
    let item = state
        .store
        .get_item(&entry.item_id)
        .await?
        .ok_or_else(|| ApiError::NotFound("item".into()))?;
    Ok(Json(json!({ "item": item, "position": entry.position, "done": false })))
}

#[derive(Debug, Deserialize)]
struct AnswerBody {
    item_id: String,
    response: String,
}

async fn answer(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<AnswerBody>,
) -> ApiResult<Json<Value>> {
    let result = service::answer(
        &state,
        &body.item_id,
        &body.response,
        Some(&id),
        state.host.as_ref(),
    )
    .await?;
    state
        .store
        .bind_session_attempt(&id, &body.item_id, &result.attempt.id)
        .await?;
    Ok(Json(serde_json::to_value(result)?))
}

async fn finish_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let rescheduled = service::finish_session(&state, &id, now_ms()).await?;
    Ok(Json(json!({ "ok": true, "skills_rescheduled": rescheduled })))
}

// ── Reads ──────────────────────────────────────────────────────────────────────

async fn list_attempts(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let attempts = state
        .store
        .list_attempts_for_subject(query.subject()?, query.limit())
        .await?;
    Ok(Json(json!({ "attempts": attempts })))
}

async fn mastery(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let report = service::mastery_report(&state, query.subject()?, now_ms()).await?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn due(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let skills = state
        .store
        .list_due_skills(query.subject_id.as_deref(), now_ms(), query.limit())
        .await?;
    Ok(Json(json!({ "due": skills })))
}

async fn trajectory(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let report = service::mastery_report(&state, query.subject()?, now_ms()).await?;
    Ok(Json(json!({ "trajectory": report.trajectory })))
}

// ── Review candidates ──────────────────────────────────────────────────────────

async fn list_candidates(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let candidates = state
        .store
        .list_candidates(
            query.subject_id.as_deref(),
            Some(CandidateStatus::Pending),
            query.limit(),
        )
        .await?;
    Ok(Json(json!({ "candidates": candidates })))
}

async fn accept_candidate(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let item = service::accept_candidate(&state, &id, now_ms()).await?;
    Ok(Json(json!({ "ok": true, "item": item })))
}

async fn reject_candidate(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(
        state
            .store
            .decide_candidate(&id, CandidateStatus::Rejected, None, now_ms())
            .await?,
        "candidate",
    )
}

// ── Settings ───────────────────────────────────────────────────────────────────

async fn get_settings(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(serde_json::to_value(state.store.get_settings().await?)?))
}

async fn put_settings(
    State(state): State<AppState>,
    Json(body): Json<TuitionSettings>,
) -> ApiResult<Json<Value>> {
    state.store.put_settings(&body).await?;
    Ok(Json(json!({ "ok": true })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest, parsed from the package tree.
    fn manifest() -> Value {
        let raw = include_str!("../../manifest.json");
        serde_json::from_str(raw).expect("the manifest must be valid JSON")
    }

    fn declared_routes() -> Vec<String> {
        manifest()["sidecars"][0]["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
            .iter()
            .map(|r| r["path"].as_str().expect("a path").to_owned())
            .collect()
    }

    #[test]
    fn every_served_route_is_declared_in_the_manifest() {
        // Core's ext-proxy matches declared routes with an EXACT segment count, so an
        // undeclared path is a hard 404 that reads like a bug in this file rather than
        // a missing line in a JSON document three directories away.
        let declared = declared_routes();
        for route in SERVED_ROUTES {
            assert!(
                declared.iter().any(|d| d == route),
                "'{route}' is served but not declared in manifest.json"
            );
        }
    }

    #[test]
    fn every_declared_route_is_actually_served() {
        // The other direction. A declared route with no handler is a 404 the manifest
        // promises will work, which is worse — a workflow author reads the manifest.
        for route in declared_routes() {
            assert!(
                SERVED_ROUTES.contains(&route.as_str()),
                "'{route}' is declared in manifest.json but nothing serves it"
            );
        }
    }

    #[test]
    fn the_manifest_port_and_mount_match_the_process_constants() {
        // Four constants must agree with the manifest and nothing checks them at
        // compile time. Two of them are checkable from here.
        let manifest = manifest();
        assert_eq!(manifest["sidecars"][0]["port"], 8007);
        assert_eq!(manifest["sidecars"][0]["http"]["mount"], "/api/tuition");
        assert_eq!(manifest["id"], crate::state::PLUGIN_ID);
    }

    #[test]
    fn every_hook_event_id_is_namespaced_to_this_manifest() {
        let manifest = manifest();
        let id = manifest["id"].as_str().expect("an id");
        for event in manifest["contributes"]["hook_events"]
            .as_array()
            .expect("hook_events must be an array")
        {
            let event_id = event["id"].as_str().expect("an event id");
            assert!(
                event_id.starts_with(&format!("{id}#")),
                "'{event_id}' is not namespaced to '{id}'"
            );
        }
    }

    #[test]
    fn an_explicitly_created_skill_defaults_to_active_not_proposed() {
        // A hand-created skill that defaults to `Proposed` is excluded from
        // scheduling, so it is never planned, never attempted and never becomes due —
        // it simply does not exist as far as studying is concerned.
        let default_status = |raw: Option<&str>| match raw {
            Some(r) if !r.trim().is_empty() => SkillStatus::parse(r),
            _ => SkillStatus::Active,
        };
        assert_eq!(default_status(None), SkillStatus::Active);
        assert_eq!(default_status(Some("")), SkillStatus::Active);
        assert_eq!(default_status(Some("   ")), SkillStatus::Active);
        // The ingest path still gets what it asks for.
        assert_eq!(default_status(Some("proposed")), SkillStatus::Proposed);
        assert_eq!(default_status(Some("archived")), SkillStatus::Archived);
    }

    #[test]
    fn the_limit_clamp_refuses_a_hostile_page_size() {
        let huge = SubjectQuery {
            subject_id: Some("s".into()),
            limit: Some(1_000_000),
        };
        assert_eq!(huge.limit(), MAX_LIMIT);
        let zero = SubjectQuery {
            subject_id: Some("s".into()),
            limit: Some(0),
        };
        assert_eq!(zero.limit(), 1);
        let absent = SubjectQuery {
            subject_id: Some("s".into()),
            limit: None,
        };
        assert_eq!(absent.limit(), DEFAULT_LIMIT);
    }

    #[test]
    fn a_missing_subject_id_is_a_bad_request_not_a_silent_empty_list() {
        let query = SubjectQuery {
            subject_id: None,
            limit: None,
        };
        assert!(query.subject().is_err());
        let empty = SubjectQuery {
            subject_id: Some(String::new()),
            limit: None,
        };
        assert!(empty.subject().is_err());
    }
}
