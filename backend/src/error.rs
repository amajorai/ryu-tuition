//! One error type for the whole HTTP surface, so every handler can be written as
//! `-> ApiResult<Json<T>>` and use `?` on store/serde/host calls.
//!
//! A single enum rather than per-handler `(StatusCode, Json<Value>)` tuples: this
//! app declares 24 routes in its manifest across the CRUD surface, the session
//! runner and the candidate queue. A tuple-returning convention makes every one of
//! those handlers re-implement its own error mapping, which is exactly how a 500
//! ends up leaking a SQL string to the frame. Funnelling through one
//! `IntoResponse` gives a single place where the status code, the stable
//! machine-readable `code`, and the message-vs-detail split are decided.
//!
//! Wire shape is fixed and snake_case, matching the rest of the sidecar:
//! `{ "error": "<human message>", "code": "<machine code>" }`.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Every handler in the HTTP surface returns this.
pub type ApiResult<T> = Result<T, ApiError>;

#[derive(Debug)]
pub enum ApiError {
    /// The addressed row does not exist.
    NotFound(String),
    /// The caller's payload is structurally wrong — an item whose answer key does
    /// not match its declared kind, a prerequisite edge that would close a cycle,
    /// a session budget of zero minutes.
    BadRequest(String),
    /// The row exists but is not in a state that admits this transition — deciding
    /// a review candidate that was already accepted, answering an item in a
    /// finished session. Distinct from `BadRequest` because the client's payload
    /// was fine and a retry against a different row may succeed.
    Conflict(String),
    /// A route whose handler is owned by a module that has not landed yet. Returns
    /// 501 rather than 500 so the companion (and any smoke test) can tell "not
    /// built" from "broken", and so a monitoring alert on 5xx does not fire on
    /// known gaps.
    NotImplemented(String),
    /// A dependency we do not control failed — Core's host callback, the
    /// `document.parse` provider, the network. 502, because the fault is upstream
    /// of this process.
    ///
    /// Also the honest answer when the model edges are unreachable: the
    /// deterministic spine keeps working without them, so an ingest that cannot
    /// reach a model must fail loudly rather than degrade into proposing nothing.
    Upstream(String),
    /// A dependency this route needs is not connected — today, the host bridge that
    /// carries model calls. Deliberately 503 rather than 500: nothing is broken, the
    /// app is running outside Ryu (or without the grant), and the fix is on the
    /// caller's side.
    Unavailable(String),
    /// Anything else. The `anyhow` chain is logged in full; the client gets a fixed
    /// string, because these messages contain SQL, file paths, and occasionally
    /// fragments of credentials.
    Internal(anyhow::Error),
}

impl ApiError {
    pub fn not_found(what: impl Into<String>) -> Self {
        Self::NotFound(what.into())
    }

    pub fn bad_request(why: impl Into<String>) -> Self {
        Self::BadRequest(why.into())
    }

    pub fn conflict(why: impl Into<String>) -> Self {
        Self::Conflict(why.into())
    }

    /// The marker a later module returns until its body lands. Kept as a
    /// constructor rather than a bare string so `grep -rn "not_implemented"` finds
    /// every remaining gap in one pass.
    pub fn not_implemented(what: impl Into<String>) -> Self {
        Self::NotImplemented(what.into())
    }

    pub fn upstream(what: impl Into<String>) -> Self {
        Self::Upstream(what.into())
    }

    /// The stable machine-readable discriminator. The companion branches on this,
    /// never on the human message, so the message stays free to change.
    fn code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::BadRequest(_) => "bad_request",
            Self::Conflict(_) => "conflict",
            Self::NotImplemented(_) => "not_implemented",
            Self::Upstream(_) => "upstream_error",
            Self::Unavailable(_) => "unavailable",
            Self::Internal(_) => "internal_error",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::NotImplemented(_) => StatusCode::NOT_IMPLEMENTED,
            Self::Upstream(_) => StatusCode::BAD_GATEWAY,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(m) => write!(f, "{m} not found"),
            Self::BadRequest(m) | Self::Conflict(m) | Self::Upstream(m) | Self::Unavailable(m) => {
                write!(f, "{m}")
            }
            Self::NotImplemented(m) => write!(f, "{m} is not implemented yet"),
            Self::Internal(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Log the FULL chain before narrowing what the client sees. For `Internal`
        // this is the only place the real cause is ever recorded.
        if let Self::Internal(e) = &self {
            tracing::error!(error = ?e, "ryu-tuition: internal error");
        } else {
            tracing::debug!(error = %self, code = self.code(), "ryu-tuition: request rejected");
        }
        let status = self.status();
        let code = self.code();
        // `Internal` deliberately does NOT forward `e` — see the variant's doc.
        let message = match &self {
            Self::Internal(_) => "internal error".to_string(),
            other => other.to_string(),
        };
        (status, Json(json!({ "error": message, "code": code }))).into_response()
    }
}

// ── `?` conversions ────────────────────────────────────────────────────────────
//
// The store returns `anyhow::Result`, so `From<anyhow::Error>` is what makes every
// handler's `?` work. The `rusqlite`/`serde_json` conversions exist so a module
// that touches those crates directly (the candidate drain decoding a KV payload,
// the MCP server running its own query) does not have to
// `.map_err(anyhow::Error::from)` at each call site.

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self::Internal(e)
    }
}

impl From<rusqlite::Error> for ApiError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Internal(anyhow::Error::from(e))
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        Self::Internal(anyhow::Error::from(e))
    }
}

impl From<reqwest::Error> for ApiError {
    fn from(e: reqwest::Error) -> Self {
        // A transport failure to Core's host callback or a parse provider is
        // upstream, not our bug.
        Self::Upstream(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_and_codes_are_stable() {
        assert_eq!(ApiError::not_found("skill").status(), StatusCode::NOT_FOUND);
        assert_eq!(ApiError::not_found("skill").code(), "not_found");
        assert_eq!(
            ApiError::bad_request("cycle").status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(ApiError::conflict("decided").status(), StatusCode::CONFLICT);
        assert_eq!(
            ApiError::not_implemented("trajectory").status(),
            StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(ApiError::upstream("parse").status(), StatusCode::BAD_GATEWAY);
        assert_eq!(ApiError::upstream("parse").code(), "upstream_error");
    }

    #[test]
    fn internal_errors_do_not_leak_their_cause_to_the_client() {
        let err = ApiError::Internal(anyhow::anyhow!(
            "UPDATE skills SET mastery = ?1 failed: /Users/me/.ryu/tuition.db is locked"
        ));
        let message = match &err {
            ApiError::Internal(_) => "internal error".to_string(),
            other => other.to_string(),
        };
        assert_eq!(message, "internal error");
        assert_eq!(err.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn a_rusqlite_failure_becomes_internal_not_upstream() {
        // The distinction is what keeps a monitoring rule on 502 meaning "a
        // dependency we do not control broke" rather than "our own SQL is wrong".
        let err: ApiError = rusqlite::Error::QueryReturnedNoRows.into();
        assert!(matches!(err, ApiError::Internal(_)));
        assert_eq!(err.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
