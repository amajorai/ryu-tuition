//! Calling back into Core: the side-model edge, and the KV queue the turn hook fills.
//!
//! Two callbacks, both over loopback, both authenticated with the token Core mints for
//! this sidecar at spawn:
//!
//! ```text
//! POST http://127.0.0.1:$RYU_CORE_PORT/api/host/model/complete
//! POST http://127.0.0.1:$RYU_CORE_PORT/api/host/rpc          { method, args }
//!   authorization: Bearer $RYU_EXT_TOKEN
//!   x-ryu-plugin-id: $RYU_EXT_PLUGIN_ID
//! ```
//!
//! # Why the KV queue exists
//!
//! The Study-mode turn hook runs in Core's Deno sandbox, which has **no HTTP**. It
//! cannot call this sidecar, and there is no seam that would let it — that is a
//! deliberate property of the sandbox, not an oversight to route around.
//!
//! What both sides *can* reach is `storage.*`, which is in the kernel-contracts
//! host-API table under the `storage:kv` grant. The hook writes candidate review items
//! with `host.storage.set`; this process drains them with the same methods over
//! `/api/host/rpc`. So the handoff is an asynchronous queue through Core's own KV, and
//! neither side needs to know the other's address.
//!
//! # Absent host
//!
//! [`Host::from_env`] returns `None` when any of the three environment variables is
//! missing, which is the normal state when this binary runs standalone (its own tests
//! do exactly that). Model-backed routes then answer 503 with a message that says what
//! to do, rather than an empty result that reads like the model had nothing to say.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

const ENV_TOKEN: &str = "RYU_EXT_TOKEN";
const ENV_PLUGIN_ID: &str = "RYU_EXT_PLUGIN_ID";
const ENV_CORE_PORT: &str = "RYU_CORE_PORT";

/// Prefix of the KV keys the Study-mode hook writes.
///
/// Must match `hooks/study.js` exactly. Nothing checks it at compile time — the two
/// sides are a JS fragment and a Rust binary — so it is stated in both files and in
/// the manifest's hook description.
pub const CANDIDATE_KEY_PREFIX: &str = "tuition/candidates/";

/// The host bridge, when this process is Core-hosted.
#[derive(Debug, Clone)]
pub struct Host {
    base: String,
    token: String,
    plugin_id: String,
    http: reqwest::Client,
    timeout: Duration,
}

impl Host {
    /// Build a bridge from the environment Core injects at spawn, or `None` when this
    /// process is running standalone.
    #[must_use]
    pub fn from_env(http: reqwest::Client, timeout_ms: u64) -> Option<Host> {
        let token = non_empty(ENV_TOKEN)?;
        let plugin_id = non_empty(ENV_PLUGIN_ID)?;
        let port = non_empty(ENV_CORE_PORT)?;
        Some(Host {
            base: format!("http://127.0.0.1:{port}"),
            token,
            plugin_id,
            http,
            timeout: Duration::from_millis(timeout_ms),
        })
    }

    /// One side-model completion. Requires the `hook:side-model` grant in BOTH the
    /// sidecar's `host_api.grants` and the top-level `permission_grants` — the host
    /// authorizes on declared ∩ Gateway-approved, so one alone is a runtime 403 with
    /// nothing at parse time to explain it.
    pub async fn complete(
        &self,
        system: &str,
        prompt: &str,
        model_pref_key: Option<&str>,
    ) -> Result<String> {
        let mut body = json!({ "system": system, "prompt": prompt });
        if let Some(key) = model_pref_key {
            body["model_pref_key"] = json!(key);
        }
        let response = self
            .http
            .post(format!("{}/api/host/model/complete", self.base))
            .bearer_auth(&self.token)
            .header("x-ryu-plugin-id", &self.plugin_id)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .context("the model callback to Core failed")?;
        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .context("Core's model response was not JSON")?;
        if !status.is_success() {
            let detail = payload
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("no detail");
            return Err(anyhow!("Core refused the model call ({status}): {detail}"));
        }
        payload
            .get("text")
            .or_else(|| payload.get("content"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("Core's model response carried no text"))
    }

    /// One host-API method over `/api/host/rpc`.
    pub async fn rpc(&self, method: &str, args: Value) -> Result<Value> {
        let response = self
            .http
            .post(format!("{}/api/host/rpc", self.base))
            .bearer_auth(&self.token)
            .header("x-ryu-plugin-id", &self.plugin_id)
            .timeout(self.timeout)
            .json(&json!({ "method": method, "args": args }))
            .send()
            .await
            .with_context(|| format!("the '{method}' host call failed"))?;
        let status = response.status();
        let payload: Value = response
            .json()
            .await
            .with_context(|| format!("Core's '{method}' response was not JSON"))?;
        if !status.is_success() {
            let detail = payload
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("no detail");
            return Err(anyhow!("Core refused '{method}' ({status}): {detail}"));
        }
        Ok(payload.get("result").cloned().unwrap_or(payload))
    }

    /// Drain the Study-mode hook's candidate queue.
    ///
    /// Read-then-delete per key rather than a bulk operation, because a bulk delete
    /// that raced a hook write would drop a candidate the app never saw. Deleting only
    /// keys whose value was successfully read means a value this build cannot parse is
    /// LEFT in place for a later build rather than silently discarded.
    pub async fn drain_candidates(&self, limit: usize) -> Result<Vec<(String, Value)>> {
        let listed = self
            .rpc("storage.keys", json!({ "prefix": CANDIDATE_KEY_PREFIX }))
            .await?;
        let keys: Vec<String> = listed
            .get("keys")
            .or(Some(&listed))
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|k| k.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();

        let mut drained = Vec::new();
        for key in keys.into_iter().take(limit) {
            let value = match self.rpc("storage.get", json!({ "key": key })).await {
                Ok(value) => value,
                Err(err) => {
                    // Leave the key. A transient read failure must not consume it.
                    tracing::warn!(key, error = %err, "tuition: candidate read failed; leaving it queued");
                    continue;
                }
            };
            let inner = value.get("value").cloned().unwrap_or(value);
            if inner.is_null() {
                // An empty slot: delete it so it stops being listed.
                let _ = self.rpc("storage.delete", json!({ "key": key })).await;
                continue;
            }
            if let Err(err) = self.rpc("storage.delete", json!({ "key": key })).await {
                // Read succeeded but delete failed: skip it rather than filing it,
                // or the next tick files the same candidate again.
                tracing::warn!(key, error = %err, "tuition: candidate delete failed; skipping to avoid a duplicate");
                continue;
            }
            drained.push((key, inner));
        }
        Ok(drained)
    }
}

fn non_empty(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

/// The message a model-backed route returns when there is no host.
///
/// Actionable on purpose: "unavailable" sends someone to the logs, this sends them to
/// the right place.
#[must_use]
pub fn no_host_message() -> &'static str {
    "this needs a model, and the app is not connected to Ryu right now — \
     open it from the Ryu desktop app rather than running the binary directly"
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize the env-var mutations: these tests share one process environment and
    /// `cargo test` runs them on separate threads.
    fn with_env<T>(vars: &[(&str, Option<&str>)], body: impl FnOnce() -> T) -> T {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let saved: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(k, _)| ((*k).to_owned(), std::env::var(k).ok()))
            .collect();
        for (key, value) in vars {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        let out = body();
        for (key, value) in saved {
            match value {
                Some(v) => std::env::set_var(&key, v),
                None => std::env::remove_var(&key),
            }
        }
        out
    }

    #[test]
    fn a_standalone_process_has_no_host_rather_than_a_broken_one() {
        // Running outside Core is a SUPPORTED state — this crate's own tests do it —
        // so it must not be a startup error, and it must not produce a bridge that
        // builds doomed requests.
        with_env(
            &[
                (ENV_TOKEN, None),
                (ENV_PLUGIN_ID, None),
                (ENV_CORE_PORT, None),
            ],
            || {
                assert!(Host::from_env(reqwest::Client::new(), 1000).is_none());
            },
        );
    }

    #[test]
    fn a_partially_configured_host_is_no_host_at_all() {
        // Two of three set is not "mostly working": every call would 401. Failing to
        // build the bridge turns that into one clear 503 instead.
        with_env(
            &[
                (ENV_TOKEN, Some("t")),
                (ENV_PLUGIN_ID, Some("@ryu/tuition")),
                (ENV_CORE_PORT, None),
            ],
            || assert!(Host::from_env(reqwest::Client::new(), 1000).is_none()),
        );
    }

    #[test]
    fn whitespace_only_values_do_not_count_as_configured() {
        with_env(
            &[
                (ENV_TOKEN, Some("   ")),
                (ENV_PLUGIN_ID, Some("@ryu/tuition")),
                (ENV_CORE_PORT, Some("8080")),
            ],
            || assert!(Host::from_env(reqwest::Client::new(), 1000).is_none()),
        );
    }

    #[test]
    fn a_fully_configured_host_builds_a_loopback_base() {
        with_env(
            &[
                (ENV_TOKEN, Some("t")),
                (ENV_PLUGIN_ID, Some("@ryu/tuition")),
                (ENV_CORE_PORT, Some("8980")),
            ],
            || {
                let host = Host::from_env(reqwest::Client::new(), 1000).expect("configured");
                // Loopback only. A sidecar must never be reachable off-box.
                assert_eq!(host.base, "http://127.0.0.1:8980");
            },
        );
    }

    #[test]
    fn the_candidate_key_prefix_is_namespaced_to_this_app() {
        // The hook and this binary agree on this string with nothing checking it, so
        // at minimum it must be scoped so a collision with another app is impossible.
        assert!(CANDIDATE_KEY_PREFIX.starts_with("tuition/"));
        assert!(CANDIDATE_KEY_PREFIX.ends_with('/'));
    }

    #[test]
    fn the_no_host_message_tells_the_reader_what_to_do() {
        let message = no_host_message();
        assert!(message.contains("Ryu"), "{message}");
        assert!(message.len() > 40, "too terse to act on: {message}");
    }
}
