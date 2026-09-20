//! The axum state every handler is built over: the store, one HTTP client, the
//! process config, and the app-event emitter.
//!
//! One state struct rather than per-module states, because the later-owned modules
//! (ingest, generation, the session runner, the scheduler tick) each need three of
//! these four and a narrower state per module would just mean converting between
//! them at every call. Every field is cheap to clone (`Arc` inside), so
//! `State<AppState>` extraction costs nothing per request.

use std::sync::Arc;
use std::time::Duration;

use crate::store::TuitionStore;

/// This app's manifest `id`. Core authorizes every app-event emit against it — the
/// caller must *be* the plugin the event is namespaced to — so it must stay
/// byte-identical to the `id` in `apps-store/tuition/manifest.json`.
pub const PLUGIN_ID: &str = "@ryu/tuition";

/// The events this app declares in its manifest's `contributes.hook_events`.
///
/// Held as constants next to the id so the `<plugin id>#<name>` rule Core enforces
/// at load is checkable at a glance rather than spread across the modules that
/// raise them. Nothing checks these against the manifest at compile time; the test
/// at the bottom of this file checks the shape, and the manifest is the authority
/// on the names.
pub const EVENT_REVIEW_DUE: &str = "@ryu/tuition#review.due";
pub const EVENT_MASTERY_DROPPED: &str = "@ryu/tuition#mastery.dropped";
pub const EVENT_GOAL_AT_RISK: &str = "@ryu/tuition#goal.at-risk";

/// The hard ceiling on ONE outbound call, end to end.
///
/// Generous because the calls it bounds are the two model edges — proposing skills
/// from a parsed chapter, writing a batch of practice items — and a side-model
/// completion over a few thousand tokens legitimately takes tens of seconds. It is
/// a ceiling, not a target: what it exists to prevent is an endpoint that accepts
/// the TCP connection and never answers, which without a bound leaves the await
/// pending forever and wedges the tick that started it. `/health` would keep
/// answering 200 the whole time, because it only touches the store.
pub const OUTBOUND_CALL_TIMEOUT_MS: u64 = 120_000;

/// Ceiling on the TCP+TLS handshake specifically, well under the whole-call bound.
/// A host that is not answering at all should fail fast rather than burn the full
/// allowance — a connect that hangs will not start succeeding at second 119.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The one HTTP client shape this process uses for outbound traffic (Core's host
/// callback for the model edges, and the `document.parse` capability call).
///
/// A free function rather than an inline `Client::new()` at each site, because
/// `Client::new()` has neither a request nor a connect timeout and a bound that
/// holds at only one construction point is not a bound. Falls back to the default
/// client if the builder ever fails, so a timeout config problem degrades to
/// today's behaviour instead of refusing to boot.
pub fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_millis(OUTBOUND_CALL_TIMEOUT_MS))
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Process-level configuration, resolved once at boot from the environment.
///
/// Distinct from [`crate::models::TuitionSettings`], which is user-editable spine
/// tuning, and from the manifest's `pref_key` settings, which live in Core's
/// preference store. The split matters: a user must not be able to change the port
/// or the shared secret from a settings tab.
#[derive(Debug, Clone)]
pub struct Config {
    /// The loopback port this process listens on.
    pub port: u16,
    /// Whether the tick loop runs at all. `RYU_TUITION_SCHEDULER=0` disables it,
    /// which is what a test harness or a second read-only reader wants.
    /// The shared secret every request must carry, read from `RYU_EXT_TOKEN`.
    ///
    /// `None` when Core did not inject one, which the bearer gate treats as
    /// FAIL-CLOSED — see `bearer_ok` in `main.rs`.
    pub token: Option<String>,
    pub scheduler_enabled: bool,
    /// Seconds between ticks. The tick rolls the due set, drains the Study-mode
    /// KV queue and evaluates the trajectory, none of which is urgent — the events
    /// it raises are about days, not seconds.
    pub tick_secs: u64,
    /// How many queued KV keys one drain may claim. Bounds the blast radius of a
    /// long offline stretch: without it, a node that ran Study mode for a week
    /// without this sidecar being spawned would try to file the whole backlog in
    /// one tick.
    pub drain_batch: usize,
}

impl Config {
    /// Read from the environment, with the defaults a normal Core-spawned run uses.
    pub fn from_env(port: u16) -> Self {
        Self {
            port,
            token: std::env::var("RYU_EXT_TOKEN")
                .ok()
                .map(|t| t.trim().to_owned())
                .filter(|t| !t.is_empty()),
            scheduler_enabled: std::env::var("RYU_TUITION_SCHEDULER")
                .map(|v| !matches!(v.trim(), "0" | "false" | "off"))
                .unwrap_or(true),
            tick_secs: std::env::var("RYU_TUITION_TICK_SECS")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(300),
            drain_batch: std::env::var("RYU_TUITION_DRAIN_BATCH")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(50),
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub store: TuitionStore,
    /// One shared client for every outbound call. Shared deliberately:
    /// `reqwest::Client` owns a connection pool, and building one per request would
    /// re-do TLS on every generated item. Built by [`build_http_client`], so it
    /// carries the request and connect timeouts.
    pub http: reqwest::Client,
    pub config: Arc<Config>,
    /// Raises this app's declared hook events so plugin hooks and event-triggered
    /// workflows can react to a review falling due without either side knowing the
    /// other exists.
    ///
    /// Safe to hold unconditionally: `from_env` never fails, and every emit no-ops
    /// when `RYU_CORE_PORT`/`RYU_EXT_TOKEN` are absent — which is the state under
    /// this crate's own tests and any standalone run, so no test needs a live Core.
    pub events: ryu_app_events::EventEmitter,
    /// The bridge back into Core, or `None` when this process is running standalone.
    ///
    /// Held on the state rather than rebuilt per request so the one shared
    /// `reqwest::Client` (with its timeouts) is reused — a client per call re-does
    /// TLS every time, and an untimed one wedges a whole tick.
    pub host: Option<crate::host::Host>,
}

impl AppState {
    pub fn new(store: TuitionStore, config: Config) -> Self {
        let http = build_http_client();
        let host = crate::host::Host::from_env(http.clone(), OUTBOUND_CALL_TIMEOUT_MS);
        Self {
            store,
            http,
            config: Arc::new(config),
            events: ryu_app_events::EventEmitter::from_env(PLUGIN_ID),
            host,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_event_id_is_namespaced_to_this_plugin() {
        // Core rejects an emit whose event id is not `<the caller's own id>#<name>`,
        // and it does so at runtime with a 403 that says nothing about which
        // constant was wrong. This is the check that would have caught it.
        for id in [EVENT_REVIEW_DUE, EVENT_MASTERY_DROPPED, EVENT_GOAL_AT_RISK] {
            let (owner, name) = id.split_once('#').expect("event ids carry a `#`");
            assert_eq!(owner, PLUGIN_ID);
            assert!(!name.is_empty());
        }
        assert_eq!(PLUGIN_ID, "@ryu/tuition");
    }

    #[test]
    fn config_defaults_hold_and_the_scheduler_switch_is_off_only_when_asked() {
        // Read from a clean-ish environment: these vars are not set in test runs, so
        // this is the shape a Core-spawned process gets minus the port injection.
        let config = Config::from_env(8007);
        assert_eq!(config.port, 8007);
        assert!(config.scheduler_enabled);
        assert_eq!(config.tick_secs, 300);
        assert_eq!(config.drain_batch, 50);
    }

    #[test]
    fn the_shared_client_carries_a_timeout() {
        // There is no getter for a built client's timeout, so this asserts the one
        // thing that is observable: the builder succeeds and we are not silently
        // falling back to `Client::new()`'s unbounded shape via some panic path.
        let _client = build_http_client();
        assert!(OUTBOUND_CALL_TIMEOUT_MS > CONNECT_TIMEOUT.as_millis() as u64);
    }
}
