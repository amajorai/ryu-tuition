//! The MCP server: five tools, so any agent can tutor you and record the attempt.
//!
//! This is what makes the app part of Ryu rather than a page inside it. A chat, a
//! subagent or a workflow `mcp` node can ask what is due, serve questions, grade an
//! answer and read the mastery report — and every one of those calls goes through
//! [`crate::service`], the same path the companion uses. An agent drilling you cannot
//! update the mastery model differently than the app does.
//!
//! # Two things that silently break this
//!
//! - **stdout is the wire.** Every log line must go to stderr. One `println!` — or one
//!   `tracing` subscriber left on the default writer — desynchronizes JSON-RPC framing
//!   and every later frame is discarded by the client, with no error anywhere.
//! - **Tool names are registered BARE.** Core forms the id as `{server}__{tool}` from
//!   the `mcp_servers` key in the manifest, so registering `tuition__due` here would
//!   produce `tuition__tuition__due`. The manifest's `provides[].tools` map carries the
//!   full ids; this file carries the short ones.
//!
//! The protocol is hand-rolled newline-delimited JSON-RPC 2.0 — no MCP SDK dependency,
//! matching `@ryu/reasoning` and `@ryu/social`.

use std::io::Write as _;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt as _, BufReader};

use crate::{host::Host, models::now_ms, service, state::AppState};

/// The protocol version this server speaks.
const PROTOCOL_VERSION: &str = "2024-11-05";

/// Default number of items a `quiz` call serves when the caller does not say.
const DEFAULT_QUIZ_MINUTES: u32 = 10;

/// The tool table.
///
/// Every description is written for a MODEL choosing between tools, not for a human
/// reading docs — it says when to reach for this one and what it will not do.
fn tools() -> Value {
    json!([
        {
            "name": "due",
            "description": "List the skills whose spaced-repetition review is due now for a subject, \
                weakest first. Read-only: it records nothing and does not start a session. Use this \
                to answer 'what should I study' before serving any questions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject_id": { "type": "string", "description": "The subject to look at. Omit to cover every subject." },
                    "limit": { "type": "integer", "description": "Maximum skills to return (default 20)." }
                }
            }
        },
        {
            "name": "quiz",
            "description": "Plan a study session for a time budget and return the questions to ask, in \
                order, with their choices. Picks the items with the best expected mastery gain per \
                minute and caps how many come from any one skill. Returns the questions ONLY — it does \
                not grade them; pass each answer to `grade`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject_id": { "type": "string" },
                    "minutes": { "type": "integer", "description": "Time budget in minutes (default 10)." }
                },
                "required": ["subject_id"]
            }
        },
        {
            "name": "grade",
            "description": "Grade one answer and fold it into the learner's mastery estimate. Multiple \
                choice, cloze, numeric and exact-match answers are decided by comparison with NO model \
                involved; only free-response answers are marked by a model, against the item's written \
                rubric, which is returned with the mark. Returns the verdict, the mastery before and \
                after, and whether the answer moved the estimate at all.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "item_id": { "type": "string" },
                    "response": { "type": "string", "description": "The learner's answer. For a cloze item, separate the blanks with '|'." },
                    "session_id": { "type": "string", "description": "Attach the attempt to a session, if one is open." }
                },
                "required": ["item_id", "response"]
            }
        },
        {
            "name": "log",
            "description": "Record that a skill was practised outside the app — a past paper, a lesson, \
                a conversation — without grading a specific stored item. Moves the mastery estimate the \
                same way a graded attempt would. Use this when you taught something and want it to count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_id": { "type": "string" },
                    "correct": { "type": "boolean", "description": "Whether the learner got it right." },
                    "note": { "type": "string", "description": "What was practised, for the history." }
                },
                "required": ["skill_id", "correct"]
            }
        },
        {
            "name": "mastery",
            "description": "The mastery report for a subject: the posterior probability the learner \
                knows each skill, how many more correct answers would reach their target, and — when the \
                subject has an exam date and at least three finished sessions — whether they are on \
                track. Reports 'unknown' rather than guessing below that. Read-only.",
            "inputSchema": {
                "type": "object",
                "properties": { "subject_id": { "type": "string" } },
                "required": ["subject_id"]
            }
        }
    ])
}

/// Serve the MCP protocol on stdin/stdout until the client closes the stream.
pub async fn serve(state: AppState, host: Option<Host>) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // An unparseable frame is skipped rather than answered: without an id there is
        // nothing to answer, and guessing one corrupts the client's pending map.
        let Ok(request) = serde_json::from_str::<Value>(line) else {
            tracing::warn!("mcp: skipping a frame that was not JSON");
            continue;
        };
        let Some(id) = request.get("id").cloned() else {
            // A notification. Nothing to reply to.
            continue;
        };
        let method = request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let params = request.get("params").cloned().unwrap_or_else(|| json!({}));

        let response = match method {
            "initialize" => ok(
                id,
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "tuition", "version": env!("CARGO_PKG_VERSION") }
                }),
            ),
            "ping" => ok(id, json!({})),
            "tools/list" => ok(id, json!({ "tools": tools() })),
            "tools/call" => {
                let name = params
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match call(&state, host.as_ref(), &name, &args).await {
                    Ok(value) => ok(id, content(&value, false)),
                    // A tool failure is a RESULT with `isError`, not a JSON-RPC error:
                    // the model needs to read what went wrong and try something else,
                    // and a protocol error is not shown to it.
                    Err(err) => ok(id, content(&json!({ "error": err.to_string() }), true)),
                }
            }
            other => err(id, -32601, &format!("unknown method '{other}'")),
        };
        emit(&response);
    }
    Ok(())
}

fn ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn err(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn content(value: &Value, is_error: bool) -> Value {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

/// Write one frame and FLUSH.
///
/// The flush is not optional: stdout to a pipe is block-buffered, so without it the
/// client waits forever for a reply that is sitting in this process's buffer.
fn emit(frame: &Value) {
    let mut stdout = std::io::stdout().lock();
    if writeln!(stdout, "{frame}").is_ok() {
        let _ = stdout.flush();
    }
}

async fn call(state: &AppState, host: Option<&Host>, name: &str, args: &Value) -> Result<Value> {
    let now = now_ms();
    let subject = |key: &str| -> Option<String> {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|s| !s.is_empty())
    };

    match name {
        "due" => {
            let limit = args
                .get("limit")
                .and_then(Value::as_i64)
                .unwrap_or(20)
                .clamp(1, 200);
            let skills = state
                .store
                .list_due_skills(subject("subject_id").as_deref(), now, limit)
                .await?;
            Ok(json!({
                "due": skills.iter().map(|s| json!({
                    "skill_id": s.id,
                    "subject_id": s.subject_id,
                    "name": s.name,
                    "mastery": s.mastery,
                    "due_at": s.due_at,
                    "lapses": s.lapses,
                })).collect::<Vec<_>>()
            }))
        }
        "quiz" => {
            let subject_id =
                subject("subject_id").ok_or_else(|| anyhow::anyhow!("subject_id is required"))?;
            let minutes = args
                .get("minutes")
                .and_then(Value::as_u64)
                .unwrap_or(u64::from(DEFAULT_QUIZ_MINUTES))
                .clamp(1, 240) as u32;
            let planned = service::plan_session(state, &subject_id, minutes, now).await?;
            let mut questions = Vec::new();
            for entry in &planned {
                if let Some(item) = state.store.get_item(&entry.item_id).await? {
                    questions.push(json!({
                        "item_id": item.id,
                        "skill_id": item.skill_id,
                        "kind": item.kind.as_str(),
                        "prompt": item.prompt,
                        "choices": item.choices,
                        "est_seconds": entry.est_cost_ms / 1000,
                    }));
                }
            }
            Ok(json!({ "questions": questions, "planned_minutes": minutes }))
        }
        "grade" => {
            let item_id = args
                .get("item_id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("item_id is required"))?;
            let response = args
                .get("response")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("response is required"))?;
            let session_id = args.get("session_id").and_then(Value::as_str);
            let result = service::answer(state, item_id, response, session_id, host).await?;
            Ok(serde_json::to_value(result)?)
        }
        "log" => {
            let skill_id = args
                .get("skill_id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("skill_id is required"))?;
            let correct = args
                .get("correct")
                .and_then(Value::as_bool)
                .ok_or_else(|| anyhow::anyhow!("correct is required"))?;
            let note = args.get("note").and_then(Value::as_str);
            let result = service::log_practice(state, skill_id, correct, note).await?;
            Ok(serde_json::to_value(result)?)
        }
        "mastery" => {
            let subject_id =
                subject("subject_id").ok_or_else(|| anyhow::anyhow!("subject_id is required"))?;
            let report = service::mastery_report(state, &subject_id, now).await?;
            Ok(serde_json::to_value(report)?)
        }
        other => Err(anyhow::anyhow!("unknown tool '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tool_name_is_self_prefixed() {
        // Core forms `{server}__{tool}`, so a name carrying the server prefix here
        // becomes `tuition__tuition__due` and no caller can ever reach it.
        for tool in tools().as_array().expect("an array") {
            let name = tool["name"].as_str().expect("a name");
            assert!(
                !name.contains("__"),
                "'{name}' is self-prefixed; Core adds the server prefix"
            );
            assert!(
                !name.starts_with("tuition"),
                "'{name}' repeats the server name"
            );
        }
    }

    #[test]
    fn every_tool_has_an_object_schema_and_a_real_description() {
        // A model picks a tool by its description. A terse one gets the tool called
        // in the wrong situation, which for `grade` means a recorded attempt that
        // moves the learner's mastery on a question they were never asked.
        for tool in tools().as_array().expect("an array") {
            let name = tool["name"].as_str().expect("a name");
            let description = tool["description"].as_str().unwrap_or_default();
            assert!(
                description.len() > 40,
                "'{name}' has a {}-character description",
                description.len()
            );
            assert_eq!(
                tool["inputSchema"]["type"], "object",
                "'{name}' must take an object"
            );
        }
    }

    #[test]
    fn the_tool_set_matches_what_the_manifest_advertises() {
        // The manifest's `provides[].tools` map is the capability vocabulary; this
        // table is what actually answers. They are edited in different files, so a
        // rename in one is invisible in the other.
        // Bound to a local: `tools()` returns an owned Value, and borrowing names
        // out of a temporary would drop it at the end of the statement.
        let table = tools();
        let mut names: Vec<&str> = table
            .as_array()
            .expect("an array")
            .iter()
            .map(|t| t["name"].as_str().expect("a name"))
            .collect();
        names.sort_unstable();
        assert_eq!(names, ["due", "grade", "log", "mastery", "quiz"]);
    }

    #[test]
    fn a_tool_failure_is_a_result_not_a_protocol_error() {
        // A JSON-RPC error is not shown to the model; an `isError` result is. Getting
        // this backwards means the model sees a silent nothing and retries forever.
        let frame = content(&json!({ "error": "no such item" }), true);
        assert_eq!(frame["isError"], true);
        assert!(frame["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .contains("no such item"));
    }

    #[test]
    fn responses_carry_the_jsonrpc_envelope() {
        let frame = ok(json!(7), json!({ "ok": true }));
        assert_eq!(frame["jsonrpc"], "2.0");
        assert_eq!(frame["id"], 7);
        let failure = err(json!("abc"), -32601, "unknown method 'x'");
        assert_eq!(failure["error"]["code"], -32601);
        assert_eq!(failure["id"], "abc");
    }
}
