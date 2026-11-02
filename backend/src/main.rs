//! Process shell for `ryu-tuition`.
//!
//! Three modes, decided before anything binds a port:
//!
//! - `ryu-tuition mcp` — serve the MCP protocol on stdin/stdout and exit. **Nothing
//!   binds a listener in this mode**, so several MCP clients can run their own copy.
//! - `ryu-tuition` — serve HTTP on loopback, behind the bearer gate.
//! - either — the store is opened and migrated first, so a schema problem is a
//!   startup failure rather than a 500 on the first request.
//!
//! # The four constants
//!
//! Two of these must match `apps-store/tuition/manifest.json` byte-for-byte and
//! **nothing checks them at compile time** (the other two, the plugin id and the
//! event ids, live in `state.rs` and are checked there):
//!
//! - [`DEFAULT_PORT`] == `sidecars[0].port`. Drift means Core probes a port nobody is
//!   listening on and reports the sidecar unhealthy while it happily serves.
//! - [`MOUNT`] == `sidecars[0].http.mount`. Drift means Core forwards
//!   `/api/tuition/skills` to a router that only knows `/skills`.
//!
//! `api.rs` asserts both against the manifest, which is the only reason they are safe.
//!
//! # No graceful shutdown, deliberately
//!
//! Core owns this process's lifecycle. What the shutdown path DOES do is abort the
//! background tick before returning, so a supervised restart never briefly runs two
//! ticks against one database.

use std::net::{Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::Response,
    routing::get,
    Json, Router,
};
use serde_json::json;
use tracing_subscriber::EnvFilter;

use ryu_tuition::{
    api, mcp, paths,
    seal_keys::PluginKeys,
    sidecar_link,
    state::{AppState, Config},
    store::{TuitionStore, PLUGIN_ID},
    tick,
};

/// Loopback port. 8004/8005/8006 are taken by `@ryu/ugc`, `@ryu/social` and
/// `@ryu/reasoning`; there is no port registry, so this was picked by reading the
/// other manifests. Core injects `RYU_TUITION_PORT` (profile-shifted) at spawn and
/// this is only the standalone fallback.
const DEFAULT_PORT: u16 = 8007;

/// Everything this process serves lives under here, matching the manifest's
/// `http.mount`. `/health` is the one route outside it.
const MOUNT: &str = "/api/tuition";

/// The shared secret Core mints for this sidecar.
const ENV_TOKEN: &str = "RYU_EXT_TOKEN";

fn main() -> Result<()> {
    // Take the sealing key Core delivered out of the environment before any
    // thread starts (the async runtime spawns workers).
    ryu_tuition::seal_keys::capture()?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<()> {
    // stderr, always: in `mcp` mode stdout carries the JSON-RPC stream, and one log
    // line on stdout desynchronizes framing for every later frame.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let port: u16 = std::env::var("RYU_TUITION_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let data_dir = paths::data_dir();
    let db_path = data_dir.join(ryu_tuition::store::STORE);
    let mcp_mode = std::env::args().nth(1).as_deref() == Some("mcp");
    // The sidecar resolves its key normally. The stdio MCP process gets no key
    // from Core and never generates a standalone root (it would seal under a
    // key the sidecar does not hold); it runs its tools through the sidecar.
    let keys = if mcp_mode {
        PluginKeys::resolve_existing(PLUGIN_ID, &data_dir)?
    } else {
        PluginKeys::resolve(PLUGIN_ID, &data_dir)?
    };
    let store = TuitionStore::open_with_keys(db_path.clone(), &keys)
        .with_context(|| format!("opening {}", db_path.display()))?;
    let state = AppState::new(store, Config::from_env(port));

    // Dispatched AFTER the state exists and BEFORE any listener binds.
    if mcp_mode {
        let host = state.host.clone();
        return mcp::serve(state, host, Some(data_dir)).await;
    }
    let handoff_state = state.clone();

    let ticker = if state.config.scheduler_enabled {
        Some(tokio::spawn(tick::run(state.clone())))
    } else {
        tracing::info!("tuition: the review tick is disabled by configuration");
        None
    };

    // The gated half: everything under the mount, behind the bearer middleware.
    // `api::routes` already applied the state, so this is a `Router<()>`.
    //
    // `/openapi.json` rides INSIDE that same gate, at the SERVER ROOT. Core fetches
    // `http://127.0.0.1:<port>/openapi.json` on this sidecar's first Healthy edge and
    // lowers every operation it finds into a searchable LLM tool, so routing this one
    // endpoint is what makes the whole `/api/tuition` surface callable by an agent —
    // tool derivation reads the document, never the router.
    //
    // Root, not under the mount: Core tries the root FIRST, and keeping the document
    // off the mount keeps it out of the manifest's declared `http.routes[]` — anything
    // declared there is reachable through the generic ext-proxy, and this schema is
    // Core's to read, not an app surface. Inside the gate, not next to the un-gated
    // `/health`: Core stamps the injected `RYU_EXT_TOKEN` on the fetch, so the gate
    // costs the fetcher nothing, while un-gated it would disclose this app's entire
    // internal API surface to any other process on loopback. The handler closes over
    // no state, so `gated` stays a `Router<()>` and the merge below is unaffected.
    let gated = Router::new()
        .nest(MOUNT, api::routes(state.clone()))
        .route("/openapi.json", get(|| async { Json(api::openapi()) }))
        .layer(from_fn_with_state(state.clone(), bearer_gate));

    let app = Router::new()
        // `/health` sits on the OUTER router, UNGATED: Core probes it before it has
        // any reason to trust this process, and a gated health check would report a
        // healthy sidecar as down for the whole of its first second. Its state is
        // applied here so both halves are `Router<()>` before they merge.
        .route("/health", get(health))
        .with_state(state)
        .merge(gated);

    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    // Publish the loopback handoff for the keyless stdio `mcp` process.
    let handoff = sidecar_link::publish(&data_dir, listener.local_addr()?.port())?;
    let app = app.merge(api::handoff_routes(handoff_state, handoff.token.clone()));
    tracing::info!(port, mount = MOUNT, db = %db_path.display(), "ryu-tuition: listening");

    let served = axum::serve(listener, app)
        .with_graceful_shutdown(sidecar_link::shutdown_signal())
        .await;
    sidecar_link::withdraw(&data_dir, &handoff);

    // Abort BEFORE returning the serve result, so a supervised restart cannot briefly
    // run two ticks against one database.
    if let Some(handle) = ticker {
        handle.abort();
    }
    served.context("the HTTP server stopped")?;
    Ok(())
}

/// Liveness AND readiness: the store must answer.
///
/// A health check that only proves the process is running is worse than none — Core
/// would report a sidecar with an unreadable database as healthy and route traffic
/// into 500s.
async fn health(State(state): State<AppState>) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.store.counts().await {
        Ok(counts) => Ok(Json(json!({
            "ok": true,
            "subjects": counts.subjects,
            "skills": counts.skills,
            "items": counts.items,
            "sealing": state.store.seal_health(),
        }))),
        Err(err) => {
            tracing::error!(error = %err, "tuition: health check could not read the store");
            Err(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
}

/// Loopback (belt) plus a shared-secret bearer (suspenders).
async fn bearer_gate(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let provided = bearer_of(request.headers());
    if ryu_sidecar_runtime::token_ok(provided.as_deref(), state.config.token.as_deref()) {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

fn bearer_of(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_owned)
}

/// Whether a token is configured at all, for the startup warning.
#[must_use]
pub fn token_configured() -> bool {
    std::env::var(ENV_TOKEN)
        .ok()
        .is_some_and(|t| !t.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bearer_prefix_is_required_and_case_sensitive() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer abc".parse().expect("a valid header"),
        );
        assert_eq!(bearer_of(&headers).as_deref(), Some("abc"));

        let mut bare = HeaderMap::new();
        bare.insert(
            axum::http::header::AUTHORIZATION,
            "abc".parse().expect("a valid header"),
        );
        assert!(bearer_of(&bare).is_none());
    }

    #[test]
    fn the_mount_matches_the_manifest() {
        // One of the four constants nothing else checks. `api.rs` asserts the route
        // table against the same file; this asserts the prefix they hang off.
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON");
        assert_eq!(manifest["sidecars"][0]["http"]["mount"], MOUNT);
        assert_eq!(manifest["sidecars"][0]["port"], DEFAULT_PORT);
        assert_eq!(
            manifest["sidecars"][0]["process"]["port_env"],
            "RYU_TUITION_PORT"
        );
    }
}
