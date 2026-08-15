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
//! # …and in the OpenAPI document
//!
//! Core fetches `GET /openapi.json` from this sidecar on its first Healthy edge and
//! lowers every operation it finds into a searchable LLM tool — then keeps only the
//! ones the manifest ALSO declares. So a route with no `#[utoipa::path]` annotation
//! contributes no tool at all: nothing errors, an agent simply cannot reach it. That
//! third direction is asserted by `every_declared_route_appears_in_the_openapi_doc`.
//!
//! The annotations carry the ABSOLUTE external path in `{param}` form while the router
//! registers paths RELATIVE to the mount in axum's `:param` form. The two forms differ
//! on purpose — Core nests this router at the mount, and a caller hits the absolute
//! path. The test normalises between them; do not "align" either side.
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
        SourceKind, Tolerance, TuitionSettings,
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

/// The document Core imports, served at `GET /openapi.json` by `main.rs`.
///
/// `components(schemas(...))` is what turns each `request_body = T` into a resolvable
/// `#/components/schemas/T`: without it the operation still carries a `$ref`, but the
/// target is missing and Core's `resolve_ref` yields nothing — a derived write tool
/// with zero visible arguments. utoipa 5 also auto-collects schemas reachable from the
/// annotated paths, so these rows are belt-and-braces; they are listed explicitly
/// anyway so the registration is greppable and cannot be lost to an attribute edit.
///
/// `Choice`, `AnswerKey` and `Tolerance` are here because they are reachable only
/// TRANSITIVELY, through the two item bodies — the transitive graph is the part that
/// breaks builds. Those fields carry `#[schema(inline)]`, so nothing currently `$ref`s
/// these entries; they stay registered so dropping an `inline` cannot leave a dangling
/// pointer.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        accept_candidate,
        add_prereq,
        answer,
        create_item,
        create_skill,
        create_source,
        create_subject,
        delete_item,
        delete_skill,
        delete_source,
        delete_subject,
        due,
        finish_session,
        generate_items,
        get_item,
        get_session,
        get_settings,
        get_skill,
        get_source,
        get_subject,
        list_attempts,
        list_candidates,
        list_items,
        list_prereqs,
        list_sessions,
        list_skills,
        list_sources,
        list_subjects,
        mastery,
        next_item,
        plan_session,
        put_settings,
        reject_candidate,
        remove_prereq,
        trajectory,
        update_item,
        update_skill,
        update_subject,
    ),
    components(schemas(
        AnswerBody,
        AnswerKey,
        Choice,
        GenerateBody,
        ItemBody,
        ItemPatch,
        PlanBody,
        PrereqBody,
        SkillBody,
        SourceBody,
        SubjectBody,
        Tolerance,
        TuitionSettings,
    ))
)]
struct TuitionApiDoc;

pub fn openapi() -> utoipa::openapi::OpenApi {
    <TuitionApiDoc as utoipa::OpenApi>::openapi()
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

#[utoipa::path(
    get,
    path = "/api/tuition/subjects",
    tag = "Tuition",
    summary = "list every subject the learner is studying.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_subjects(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let subjects = state.store.list_subjects().await?;
    Ok(Json(json!({ "subjects": subjects })))
}

/// Request body for creating or replacing a subject.
///
/// The field docs are not decoration: they are lifted verbatim into the OpenAPI schema
/// and become the argument descriptions a model reads when it decides what to send.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct SubjectBody {
    /// What the learner is studying, e.g. "Organic Chemistry". Required, non-blank.
    name: String,
    /// Longer description of the subject's scope. Optional.
    detail: Option<String>,
    /// The exam this subject builds towards, as `YYYY-MM-DD`. Drives the trajectory
    /// projection and the `goal.at-risk` warning; omit if there is no exam.
    exam_date: Option<String>,
    /// IANA zone name (e.g. `Europe/Berlin`) the review day boundary is read in.
    /// Defaults to UTC. A fixed offset would shift due dates for half the year.
    timezone: Option<String>,
}

#[utoipa::path(
    post,
    path = "/api/tuition/subjects",
    tag = "Tuition",
    summary = "start studying a new subject, optionally with an exam date.",
    request_body = SubjectBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/subjects/{id}",
    tag = "Tuition",
    summary = "read one subject.",
    params(("id" = String, Path, description = "Subject id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    patch,
    path = "/api/tuition/subjects/{id}",
    tag = "Tuition",
    summary = "replace a subject's name, detail, exam date and timezone.",
    params(("id" = String, Path, description = "Subject id")),
    // Every field is written, not merged — send the whole subject, not just the part
    // that changed, or an omitted field is cleared.
    request_body = SubjectBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    delete,
    path = "/api/tuition/subjects/{id}",
    tag = "Tuition",
    summary = "delete a subject and everything studied under it.",
    params(("id" = String, Path, description = "Subject id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn delete_subject(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_subject(&id).await?, "subject")
}

// ── Sources ────────────────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/tuition/sources",
    tag = "Tuition",
    summary = "list the study material registered under one subject.",
    params(
        ("subject_id" = String, Query, description = "Subject id. Required — a request without it is a 400, not an empty list."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_sources(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let sources = state.store.list_sources(query.subject()?).await?;
    Ok(Json(json!({ "sources": sources })))
}

/// Request body for registering a study source.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct SourceBody {
    /// The subject this material belongs to. Required.
    subject_id: String,
    /// Human name for the material, e.g. "Chapter 4 — Stereochemistry".
    title: String,
    /// What the material is: `textbook`, `notes`, `slides`, `paper`, `transcript`, or
    /// `other`. Unrecognised values fall back to `other`.
    kind: Option<String>,
    /// Where it came from — a URL or file path — recorded for provenance only.
    uri: Option<String>,
    /// The full plain text to extract skills and questions from. Required and
    /// non-blank. Rich formats are converted before they reach this app; nothing
    /// here parses a PDF.
    text: String,
    /// Name of whatever produced `text`, recorded for provenance only.
    parser: Option<String>,
}

#[utoipa::path(
    post,
    path = "/api/tuition/sources",
    tag = "Tuition",
    summary = "register already-extracted study material under a subject.",
    request_body = SourceBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/sources/{id}",
    tag = "Tuition",
    summary = "read one study source, including its full text.",
    params(("id" = String, Path, description = "Source id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_source(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let source = state
        .store
        .get_source(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("source".into()))?;
    Ok(Json(serde_json::to_value(source)?))
}

#[utoipa::path(
    delete,
    path = "/api/tuition/sources/{id}",
    tag = "Tuition",
    summary = "delete one study source.",
    params(("id" = String, Path, description = "Source id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn delete_source(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_source(&id).await?, "source")
}

// ── Skills ─────────────────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/tuition/skills",
    tag = "Tuition",
    summary = "list a subject's skills with the prerequisite edges between them.",
    params(
        ("subject_id" = String, Query, description = "Subject id. Required — a request without it is a 400, not an empty list."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_skills(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let skills = state.store.list_skills(query.subject()?, None).await?;
    let edges = state.store.list_prereq_edges(query.subject()?).await?;
    Ok(Json(json!({ "skills": skills, "prereqs": edges })))
}

/// Request body for creating or replacing a skill.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct SkillBody {
    /// The subject this skill belongs to. Required on create.
    subject_id: Option<String>,
    /// What the learner should be able to do, e.g. "Assign R/S configuration".
    name: String,
    /// Longer description of what mastering it means. Optional.
    detail: Option<String>,
    /// `active` (scheduled and studied), `proposed` (awaiting review, NOT scheduled)
    /// or `archived`. Omit on create to get `active`.
    status: Option<String>,
}

#[utoipa::path(
    post,
    path = "/api/tuition/skills",
    tag = "Tuition",
    summary = "add a skill to a subject's prerequisite graph.",
    request_body = SkillBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/skills/{id}",
    tag = "Tuition",
    summary = "read one skill with its mastery posterior and its prerequisites.",
    params(("id" = String, Path, description = "Skill id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    patch,
    path = "/api/tuition/skills/{id}",
    tag = "Tuition",
    summary = "rename a skill or change whether it is scheduled.",
    params(("id" = String, Path, description = "Skill id")),
    request_body = SkillBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    delete,
    path = "/api/tuition/skills/{id}",
    tag = "Tuition",
    summary = "delete a skill and its review history.",
    params(("id" = String, Path, description = "Skill id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn delete_skill(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_skill(&id).await?, "skill")
}

#[utoipa::path(
    get,
    path = "/api/tuition/skills/{id}/prereqs",
    tag = "Tuition",
    summary = "list the skills that must be mastered before this one.",
    params(("id" = String, Path, description = "Skill id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_prereqs(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    Ok(Json(json!({ "prereqs": state.store.list_prereqs(&id).await? })))
}

/// Which edge of the prerequisite graph to add or remove.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct PrereqBody {
    /// Id of the skill that must come FIRST. Adding an edge that would close a cycle
    /// is refused with a 400 naming the path.
    prereq_id: String,
}

#[utoipa::path(
    post,
    path = "/api/tuition/skills/{id}/prereqs",
    tag = "Tuition",
    summary = "make one skill a prerequisite of another.",
    params(("id" = String, Path, description = "Id of the skill that DEPENDS on the prerequisite")),
    request_body = PrereqBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    delete,
    path = "/api/tuition/skills/{id}/prereqs",
    tag = "Tuition",
    summary = "drop a prerequisite edge between two skills.",
    params(("id" = String, Path, description = "Id of the skill that DEPENDS on the prerequisite")),
    request_body = PrereqBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/items",
    tag = "Tuition",
    summary = "list the practice questions attached to one skill.",
    params(
        ("skill_id" = String, Query, description = "Skill id. Required — a request without it is a 400, not an empty list."),
        ("include_archived" = Option<bool>, Query, description = "Include retired questions. Default false."),
        ("limit" = Option<i64>, Query, description = "Maximum questions. Default 200, clamped to 500."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

/// Request body for writing a practice question by hand.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct ItemBody {
    /// The skill this question tests. Required.
    skill_id: String,
    /// One of `mcq`, `cloze`, `numeric`, `exact`, `free`. Must agree with the `kind`
    /// inside `answer` — an `mcq` item needs an `mcq` answer key.
    kind: String,
    /// The question as the learner reads it.
    prompt: String,
    /// The selectable options, for `mcq` only. Leave empty for every other kind.
    // Inlined: Core resolves a `$ref` only at the top of a schema node, so an
    // un-inlined array-of-ref reaches the model as an opaque pointer and it cannot
    // see that a choice is `{id, text}`.
    #[serde(default)]
    #[schema(inline)]
    choices: Vec<Choice>,
    /// The gradeable answer key, tagged by `kind`. Inlined so its variants are
    /// visible: `{"kind":"mcq","choice_id":…}`, `{"kind":"cloze","blanks":[[…]]}`,
    /// `{"kind":"numeric","expected":"1.5","tolerance":{…}}`,
    /// `{"kind":"exact","text":…,"alternatives":[…]}`, `{"kind":"free","rubric":…}`.
    #[schema(inline)]
    answer: AnswerKey,
    /// Id of the study source this came from, for provenance. Optional.
    source_id: Option<String>,
    /// Where in that source, e.g. "p. 214". Optional.
    source_ref: Option<String>,
}

#[utoipa::path(
    post,
    path = "/api/tuition/items",
    tag = "Tuition",
    summary = "write one practice question by hand against a skill.",
    request_body = ItemBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/items/{id}",
    tag = "Tuition",
    summary = "read one practice question, including its answer key.",
    params(("id" = String, Path, description = "Item id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_item(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let item = state
        .store
        .get_item(&id)
        .await?
        .ok_or_else(|| ApiError::NotFound("item".into()))?;
    Ok(Json(serde_json::to_value(item)?))
}

/// Request body for correcting an existing question.
///
/// The skill and the kind are fixed at creation — a question that changes what it
/// tests is a different question, and rewriting one in place would silently rewrite
/// the history of every attempt already recorded against it.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct ItemPatch {
    /// The corrected question text.
    prompt: String,
    /// The corrected options, for `mcq` only.
    #[serde(default)]
    #[schema(inline)]
    choices: Vec<Choice>,
    /// The corrected answer key, tagged by `kind` — same shapes as on create.
    #[schema(inline)]
    answer: AnswerKey,
    /// Where in the source this came from. Optional.
    source_ref: Option<String>,
}

#[utoipa::path(
    patch,
    path = "/api/tuition/items/{id}",
    tag = "Tuition",
    summary = "correct a question's wording, options or answer key.",
    params(("id" = String, Path, description = "Item id")),
    request_body = ItemPatch,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    delete,
    path = "/api/tuition/items/{id}",
    tag = "Tuition",
    summary = "delete one practice question.",
    params(("id" = String, Path, description = "Item id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn delete_item(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    require_hit(state.store.delete_item(&id).await?, "item")
}

/// Request body for asking the model to draft questions.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct GenerateBody {
    /// The skill to write questions for. Required.
    skill_id: String,
    /// How many to draft. Default 5, clamped to 1..=20.
    count: Option<u32>,
}

#[utoipa::path(
    post,
    path = "/api/tuition/items/generate",
    tag = "Tuition",
    summary = "draft new practice questions for a skill with the model. Costs a model call.",
    request_body = GenerateBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/sessions",
    tag = "Tuition",
    summary = "list past study sessions for a subject, newest first.",
    params(
        ("subject_id" = String, Query, description = "Subject id. Required — a request without it is a 400, not an empty list."),
        ("limit" = Option<i64>, Query, description = "Maximum sessions. Default 200, clamped to 500."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

/// Request body for planning a study session.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct PlanBody {
    /// The subject to study. Required.
    subject_id: String,
    /// How long the learner has. Default 15, clamped to 1..=240. The planner fills
    /// that budget by expected mastery gain rather than by a fixed question count.
    minutes: Option<u32>,
}

#[utoipa::path(
    post,
    path = "/api/tuition/sessions/plan",
    tag = "Tuition",
    summary = "plan and start a study session that fits a time budget.",
    request_body = PlanBody,
    responses((status = 200, description = "OK — `session` is null with a `note` when nothing is due", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/sessions/{id}",
    tag = "Tuition",
    summary = "read one study session and the questions planned into it.",
    params(("id" = String, Path, description = "Session id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/sessions/{id}/next",
    tag = "Tuition",
    summary = "take the next unanswered question in a session.",
    params(("id" = String, Path, description = "Session id")),
    responses((status = 200, description = "OK — `done` is true and `item` null when the session is exhausted", body = serde_json::Value))
)]
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

/// Request body for submitting an answer.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
struct AnswerBody {
    /// The question being answered — normally the `item.id` the `next` call returned.
    item_id: String,
    /// The learner's answer, as plain text. For a multiple-choice question this is
    /// the chosen `choice.id`; for a cloze, the blanks separated by `|`.
    response: String,
}

#[utoipa::path(
    post,
    path = "/api/tuition/sessions/{id}/answer",
    tag = "Tuition",
    summary = "submit an answer; grades it and updates the skill's mastery posterior.",
    params(("id" = String, Path, description = "Session id")),
    request_body = AnswerBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    post,
    path = "/api/tuition/sessions/{id}/finish",
    tag = "Tuition",
    summary = "close a session and reschedule the skills it touched.",
    params(("id" = String, Path, description = "Session id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn finish_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let rescheduled = service::finish_session(&state, &id, now_ms()).await?;
    Ok(Json(json!({ "ok": true, "skills_rescheduled": rescheduled })))
}

// ── Reads ──────────────────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/tuition/attempts",
    tag = "Tuition",
    summary = "list recent graded attempts across a subject, newest first.",
    params(
        ("subject_id" = String, Query, description = "Subject id. Required — a request without it is a 400, not an empty list."),
        ("limit" = Option<i64>, Query, description = "Maximum attempts. Default 200, clamped to 500."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/mastery",
    tag = "Tuition",
    summary = "how well the learner knows a subject: per-skill posteriors, what is ready to learn next, and the exam trajectory.",
    params(
        ("subject_id" = String, Query, description = "Subject id. Required — a request without it is a 400, not an empty report."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn mastery(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let report = service::mastery_report(&state, query.subject()?, now_ms()).await?;
    Ok(Json(serde_json::to_value(report)?))
}

#[utoipa::path(
    get,
    path = "/api/tuition/due",
    tag = "Tuition",
    summary = "which skills are due for review right now.",
    params(
        ("subject_id" = String, Query, description = "Restrict to one subject. Omit for every subject."),
        ("limit" = Option<i64>, Query, description = "Maximum skills. Default 200, clamped to 500."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/trajectory",
    tag = "Tuition",
    summary = "whether the learner is on pace for the exam date, and by how much.",
    params(
        ("subject_id" = String, Query, description = "Subject id. Required — a request without it is a 400."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn trajectory(
    State(state): State<AppState>,
    Query(query): Query<SubjectQuery>,
) -> ApiResult<Json<Value>> {
    let report = service::mastery_report(&state, query.subject()?, now_ms()).await?;
    Ok(Json(json!({ "trajectory": report.trajectory })))
}

// ── Review candidates ──────────────────────────────────────────────────────────

#[utoipa::path(
    get,
    path = "/api/tuition/candidates",
    tag = "Tuition",
    summary = "list model-drafted questions still awaiting human review.",
    params(
        ("subject_id" = String, Query, description = "Restrict to one subject. Omit for every subject."),
        ("limit" = Option<i64>, Query, description = "Maximum candidates. Default 200, clamped to 500."),
    ),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    post,
    path = "/api/tuition/candidates/{id}/accept",
    tag = "Tuition",
    summary = "accept a drafted question into the studied item pool.",
    params(("id" = String, Path, description = "Candidate id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn accept_candidate(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let item = service::accept_candidate(&state, &id, now_ms()).await?;
    Ok(Json(json!({ "ok": true, "item": item })))
}

#[utoipa::path(
    post,
    path = "/api/tuition/candidates/{id}/reject",
    tag = "Tuition",
    summary = "reject a drafted question so it is never studied.",
    params(("id" = String, Path, description = "Candidate id")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[utoipa::path(
    get,
    path = "/api/tuition/settings",
    tag = "Tuition",
    summary = "read the tutor's thresholds and scheduling knobs.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_settings(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(serde_json::to_value(state.store.get_settings().await?)?))
}

#[utoipa::path(
    put,
    path = "/api/tuition/settings",
    tag = "Tuition",
    summary = "replace the tutor's thresholds and scheduling knobs.",
    // The whole settings object is written, not merged: read `GET /settings` first and
    // send it back changed, or an omitted field is a deserialization error.
    request_body = TuitionSettings,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

    /// The OpenAPI document as plain JSON, for the schema assertions below.
    fn doc_json() -> Value {
        serde_json::to_value(openapi()).expect("the document serializes")
    }

    /// A manifest route (relative to the mount, in axum's `:param` form) rewritten into
    /// the form the OpenAPI document uses (absolute, in `{param}` form).
    ///
    /// The two forms differ ON PURPOSE — the router registers paths relative to the
    /// mount because Core nests it there, while the `#[utoipa::path]` annotations carry
    /// the absolute EXTERNAL path a caller actually hits. Normalise here; do not
    /// "align" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    /// The resolved request-body schema for one operation, or `None` if it documents
    /// no body.
    fn request_body_schema<'a>(doc: &'a Value, path: &str, method: &str) -> Option<&'a Value> {
        let escaped = path.replace('/', "~1");
        doc.pointer(&format!(
            "/paths/{escaped}/{method}/requestBody/content/application~1json/schema"
        ))
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
    fn openapi_doc_covers_the_served_routes() {
        // The document is not dead code: `main.rs` serves it and Core fetches it to
        // derive tools, so an empty one means this app contributes nothing.
        let doc = openapi();
        assert!(!doc.paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The third direction, and the one that decides tool yield. Core's importer
        // keeps only the document operations the manifest ALSO declares, so a declared
        // route with no `#[utoipa::path]` annotation is a tool that silently never
        // exists — nothing errors, the agent simply cannot call it. (The reverse is
        // harmless: an annotated path the manifest does not declare is dropped by the
        // same filter.)
        //
        // NOTE this checks PATHS, not methods: it passes as soon as one operation
        // exists at `/api/tuition/subjects/{id}`, so it cannot catch an unannotated
        // `patch` next to an annotated `get`. Green here is necessary, not sufficient.
        let mount = manifest()["sidecars"][0]["http"]["mount"]
            .as_str()
            .expect("an http.mount")
            .to_owned();
        let doc = openapi();
        for route in declared_routes() {
            let expected = doc_path_for(&mount, &route);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{route}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    #[test]
    fn every_served_route_carries_an_operation_for_each_method_it_serves() {
        // What the path-level check above cannot see. The pairs are written out rather
        // than derived because axum's `Router` does not expose its method table either,
        // so this is the same duplication `SERVED_ROUTES` already accepts — and the
        // same payoff: a handler that loses its annotation fails here.
        let doc = doc_json();
        let methods: &[(&str, &[&str])] = &[
            ("/subjects", &["get", "post"]),
            ("/subjects/:id", &["get", "patch", "delete"]),
            ("/sources", &["get", "post"]),
            ("/sources/:id", &["get", "delete"]),
            ("/skills", &["get", "post"]),
            ("/skills/:id", &["get", "patch", "delete"]),
            ("/skills/:id/prereqs", &["get", "post", "delete"]),
            ("/items", &["get", "post"]),
            ("/items/generate", &["post"]),
            ("/items/:id", &["get", "patch", "delete"]),
            ("/sessions", &["get"]),
            ("/sessions/plan", &["post"]),
            ("/sessions/:id", &["get"]),
            ("/sessions/:id/next", &["get"]),
            ("/sessions/:id/answer", &["post"]),
            ("/sessions/:id/finish", &["post"]),
            ("/attempts", &["get"]),
            ("/mastery", &["get"]),
            ("/due", &["get"]),
            ("/trajectory", &["get"]),
            ("/candidates", &["get"]),
            ("/candidates/:id/accept", &["post"]),
            ("/candidates/:id/reject", &["post"]),
            ("/settings", &["get", "put"]),
        ];
        assert_eq!(
            methods.len(),
            SERVED_ROUTES.len(),
            "the method table must cover every served route"
        );
        for (route, verbs) in methods {
            let path = doc_path_for("/api/tuition", route);
            let escaped = path.replace('/', "~1");
            for verb in *verbs {
                assert!(
                    doc.pointer(&format!("/paths/{escaped}/{verb}")).is_some(),
                    "{verb} {path} is served but carries no #[utoipa::path] annotation"
                );
            }
        }
    }

    #[test]
    fn the_item_bodies_expose_their_answer_key_rather_than_a_pointer() {
        // The failure this app is most exposed to. `answer` is a tagged enum over a
        // second tagged enum, and Core follows a `$ref` only at the TOP of a schema
        // node — so an un-inlined `answer` reaches the model as an opaque pointer and
        // the write tools for the item surface become undiscoverable in practice while
        // still compiling, still appearing in the document, and still passing every
        // path-level check above.
        let doc = doc_json();
        for (path, method, component) in [
            ("/api/tuition/items", "post", "ItemBody"),
            ("/api/tuition/items/{id}", "patch", "ItemPatch"),
        ] {
            // The body node itself IS a `$ref` — that one is fine, it sits at the top
            // and Core follows exactly one hop. What matters is what it lands on.
            let body = request_body_schema(&doc, path, method)
                .unwrap_or_else(|| panic!("{method} {path} documents a request body"));
            assert_eq!(
                body["$ref"],
                Value::String(format!("#/components/schemas/{component}")),
                "{method} {path} must point at a resolvable component: {body:#}"
            );
            let schema = doc
                .pointer(&format!("/components/schemas/{component}"))
                .unwrap_or_else(|| panic!("{component} is registered in components"));
            let rendered = schema.to_string();
            assert!(
                !rendered.contains("$ref"),
                "{component} still carries an unresolvable $ref: {schema:#}"
            );
            // The variants and the nested tolerance are materialised, not left behind
            // a pointer two levels deep.
            for needle in ["choice_id", "rubric", "relative_percent"] {
                assert!(
                    rendered.contains(needle),
                    "{component} does not expose '{needle}': {schema:#}"
                );
            }
        }
    }

    #[test]
    fn body_less_routes_declare_no_request_body() {
        // These take only a path parameter. Documenting a body for them would invent an
        // argument the handler ignores.
        let doc = doc_json();
        for (path, method) in [
            ("/api/tuition/sessions/{id}/finish", "post"),
            ("/api/tuition/candidates/{id}/accept", "post"),
            ("/api/tuition/candidates/{id}/reject", "post"),
        ] {
            assert!(
                request_body_schema(&doc, path, method).is_none(),
                "{method} {path} must document no request body"
            );
            let escaped = path.replace('/', "~1");
            assert!(
                doc.pointer(&format!("/paths/{escaped}/{method}/parameters"))
                    .is_some(),
                "{method} {path} still documents its path parameter"
            );
        }
    }

    #[test]
    fn body_field_docs_reach_the_schema_as_argument_descriptions() {
        // Field doc comments are lifted verbatim into `description`, which is the text
        // the model actually reads when deciding how to fill an argument.
        let doc = doc_json();
        let minutes = doc
            .pointer("/components/schemas/PlanBody/properties/minutes/description")
            .and_then(Value::as_str)
            .expect("the `minutes` argument is described");
        assert!(
            minutes.contains("budget"),
            "the description must say what the number buys: {minutes}"
        );
    }

    #[test]
    fn the_required_subject_filter_is_documented_as_required() {
        // `list_skills` 400s without `subject_id`. Documenting it as optional would
        // produce a tool an agent calls empty and gets an error from, every time.
        let doc = doc_json();
        let params = doc
            .pointer("/paths/~1api~1tuition~1skills/get/parameters")
            .and_then(Value::as_array)
            .expect("the list operation documents its query parameters");
        let subject = params
            .iter()
            .find(|p| p["name"] == "subject_id")
            .expect("subject_id is documented");
        assert_eq!(subject["required"], Value::Bool(true));
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
    fn the_openapi_document_is_not_a_declared_route() {
        // It is served at the SERVER ROOT, inside the bearer gate but off the mount.
        // Declaring it would expose this app's whole internal API surface through the
        // generic ext-proxy — and would fail the two direction tests above, since
        // nothing under the mount serves it.
        assert!(!declared_routes().iter().any(|r| r.contains("openapi")));
        assert!(!SERVED_ROUTES.iter().any(|r| r.contains("openapi")));
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
