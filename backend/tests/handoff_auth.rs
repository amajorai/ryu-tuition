//! The loopback MCP handoff route runs the sealed store's tools for a keyless
//! process, so only the per-run handoff token may open it. Any other loopback
//! caller (no header, Core's ext token, a guessed token) must get 401 and the
//! tool must never run: the write tool `log` leaves no attempt behind.

use ryu_tuition::api::{handoff_routes, HANDOFF_MCP_ROUTE};
use ryu_tuition::models::{BktParams, SkillStatus};
use ryu_tuition::state::{AppState, Config};
use ryu_tuition::store::TuitionStore;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

const CORE_EXT_TOKEN: &str = "core-ext-token-for-the-mounted-api";

/// POST the handoff route over a real loopback socket; returns (status, body).
async fn post(port: u16, authorization: Option<&str>, body: &Value) -> (u16, String) {
    let payload = body.to_string();
    let mut head = format!(
        "POST {HANDOFF_MCP_ROUTE} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        payload.len()
    );
    if let Some(value) = authorization {
        head.push_str(&format!("Authorization: {value}\r\n"));
    }
    head.push_str("\r\n");
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connects");
    stream.write_all(head.as_bytes()).await.expect("writes");
    stream.write_all(payload.as_bytes()).await.expect("writes");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("reads");
    let text = String::from_utf8_lossy(&raw).into_owned();
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .expect("a status line");
    let body = text
        .split_once("\r\n\r\n")
        .map_or("", |(_, b)| b)
        .to_owned();
    (status, body)
}

#[tokio::test]
async fn only_the_handoff_token_opens_the_mcp_handoff_route() {
    let store = TuitionStore::open_in_memory().expect("opens");
    let subject = store
        .create_subject("Organic chemistry", None, None, None)
        .await
        .expect("subject");
    let skill = store
        .upsert_skill(
            &subject.id,
            "Sealed SN2 mechanisms",
            None,
            SkillStatus::Active,
            None,
            BktParams::default(),
        )
        .await
        .expect("skill");
    let mut config = Config::from_env(0);
    config.token = Some(CORE_EXT_TOKEN.to_owned());
    config.scheduler_enabled = false;
    let state = AppState::new(store.clone(), config);

    let handoff_token = "per-run-handoff-token-0123456789abcdef".to_owned();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds");
    let port = listener.local_addr().expect("addr").port();
    let app = handoff_routes(state, handoff_token.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    let call = json!({
        "name": "log",
        "arguments": { "skill_id": skill.id, "correct": true, "note": "handoff probe" }
    });
    let core_ext = format!("Bearer {CORE_EXT_TOKEN}");
    let wrong = format!("Bearer {handoff_token}x");
    let raw_token = handoff_token.clone();
    for (label, authorization) in [
        ("no Authorization", None),
        ("the Core ext token", Some(core_ext.as_str())),
        ("a wrong bearer token", Some(wrong.as_str())),
        (
            "the handoff token without the Bearer scheme",
            Some(raw_token.as_str()),
        ),
        ("an empty bearer token", Some("Bearer ")),
    ] {
        let (status, body) = post(port, authorization, &call).await;
        assert_eq!(
            status, 401,
            "{label} must be rejected, got {status}: {body}"
        );
        assert!(
            !body.contains("isError"),
            "{label} reached the tool: {body}"
        );
        let attempts = store
            .list_attempts_for_skill(&skill.id, 50)
            .await
            .expect("lists");
        assert!(
            attempts.is_empty(),
            "{label} ran the write tool: {attempts:?}"
        );
    }

    let (status, body) = post(port, Some(&format!("Bearer {handoff_token}")), &call).await;
    assert_eq!(status, 200, "{body}");
    let result: Value = serde_json::from_str(&body).expect("a JSON tool result");
    assert_eq!(result["isError"], json!(false), "{result}");
    let attempts = store
        .list_attempts_for_skill(&skill.id, 50)
        .await
        .expect("lists");
    assert_eq!(attempts.len(), 1, "the handoff token runs the tool once");
    assert_eq!(attempts[0].correct, Some(true));

    server.abort();
}
