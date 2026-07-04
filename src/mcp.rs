//! Minimal MCP (Model Context Protocol) client for batch tool execution.
//!
//! Scope is deliberately narrow (see docs/design/MCP_TOOL_INTEGRATION.md):
//! this client speaks the streamable-HTTP transport and implements only the
//! `initialize` handshake and `tools/call`. Sessions, sampling, roots, and
//! elicitation are out of scope — MCP is a call target here, not a runtime.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

const MCP_PROTOCOL_VERSION: &str = "2025-03-26";

/// Outcome of a single tool call. Failures are data, not errors: the batch
/// executor records them per-call so one bad row never poisons a batch.
#[derive(Debug, Clone)]
pub struct ToolCallOutcome {
    /// Text content returned by the tool (concatenated text blocks).
    pub content: Option<String>,
    /// True when the *tool* reported failure (MCP `isError`).
    pub is_error: bool,
    /// Transport/protocol-level error, if the call never completed.
    pub error: Option<String>,
}

impl ToolCallOutcome {
    fn transport_error(msg: impl Into<String>) -> Self {
        ToolCallOutcome {
            content: None,
            is_error: true,
            error: Some(msg.into()),
        }
    }
}

/// A connected MCP-over-HTTP session. Cheap to clone; safe to share across
/// tokio tasks (reqwest::Client is internally reference-counted).
#[derive(Debug, Clone)]
pub struct McpHttpClient {
    http: reqwest::Client,
    endpoint: String,
    session_id: Option<String>,
    next_id: std::sync::Arc<AtomicU64>,
}

impl McpHttpClient {
    /// Connect to an MCP server over streamable HTTP and run the
    /// `initialize` handshake.
    pub async fn connect(endpoint: &str, timeout: Duration) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;

        let mut client = McpHttpClient {
            http,
            endpoint: endpoint.to_string(),
            session_id: None,
            next_id: std::sync::Arc::new(AtomicU64::new(1)),
        };

        let init = client
            .post_rpc(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "polar-llama", "version": env!("CARGO_PKG_VERSION")},
                }),
            )
            .await?;

        if let Some(err) = init.rpc_error {
            return Err(format!("MCP initialize failed: {err}"));
        }
        client.session_id = init.session_id;

        // Servers expect the initialized notification before tool calls.
        client
            .post_notification("notifications/initialized", json!({}))
            .await?;

        Ok(client)
    }

    /// Invoke `tools/call`. Never returns Err: all failure modes are folded
    /// into the outcome so callers can treat results as row data.
    pub async fn call_tool(&self, name: &str, arguments: &Value) -> ToolCallOutcome {
        let response = match self
            .post_rpc("tools/call", json!({"name": name, "arguments": arguments}))
            .await
        {
            Ok(r) => r,
            Err(e) => return ToolCallOutcome::transport_error(e),
        };

        if let Some(err) = response.rpc_error {
            return ToolCallOutcome::transport_error(format!("MCP error: {err}"));
        }

        let result = match response.result {
            Some(r) => r,
            None => return ToolCallOutcome::transport_error("MCP response missing result"),
        };

        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        // Concatenate text content blocks; serialize non-text blocks as JSON
        // so nothing the server returned is silently dropped.
        let content = result.get("content").and_then(Value::as_array).map(|blocks| {
            blocks
                .iter()
                .map(|block| match block.get("type").and_then(Value::as_str) {
                    Some("text") => block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    _ => block.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        });

        // Some servers return structuredContent instead of/alongside content.
        let content = match (content, result.get("structuredContent")) {
            (Some(c), _) if !c.is_empty() => Some(c),
            (_, Some(sc)) => Some(sc.to_string()),
            (c, None) => c,
        };

        ToolCallOutcome {
            content,
            is_error,
            error: None,
        }
    }

    async fn post_rpc(&self, method: &str, params: Value) -> Result<RpcResponse, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.post_message(&body, true).await
    }

    async fn post_notification(&self, method: &str, params: Value) -> Result<(), String> {
        let body = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.post_message(&body, false).await.map(|_| ())
    }

    async fn post_message(&self, body: &Value, expect_response: bool) -> Result<RpcResponse, String> {
        let mut request = self
            .http
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(body);

        if let Some(sid) = &self.session_id {
            request = request.header("Mcp-Session-Id", sid.clone());
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        let session_id = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|v| v.to_str().ok())
            .map(String::from)
            .or_else(|| self.session_id.clone());

        let status = response.status();
        let content_type = response
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = response
            .text()
            .await
            .map_err(|e| format!("failed to read HTTP response: {e}"))?;

        if !status.is_success() {
            // 202 Accepted for notifications is covered by is_success; real
            // failures surface the body for debuggability.
            return Err(format!("HTTP {status}: {}", truncate(&text, 500)));
        }

        if !expect_response {
            return Ok(RpcResponse {
                result: None,
                rpc_error: None,
                session_id,
            });
        }

        let message = if content_type.starts_with("text/event-stream") {
            parse_sse_json_rpc(&text).ok_or_else(|| "no JSON-RPC response in SSE stream".to_string())?
        } else {
            serde_json::from_str::<Value>(&text)
                .map_err(|e| format!("invalid JSON-RPC response: {e}: {}", truncate(&text, 200)))?
        };

        Ok(RpcResponse {
            rpc_error: message.get("error").map(|e| e.to_string()),
            result: message.get("result").cloned(),
            session_id,
        })
    }
}

struct RpcResponse {
    result: Option<Value>,
    rpc_error: Option<String>,
    session_id: Option<String>,
}

/// Extract the last JSON-RPC response object from an SSE body. Streamable
/// HTTP servers may emit progress notifications first; the response to our
/// request is the event carrying an `id`.
fn parse_sse_json_rpc(body: &str) -> Option<Value> {
    let mut last_response: Option<Value> = None;
    for line in body.lines() {
        let line = line.trim_start();
        if let Some(data) = line.strip_prefix("data:") {
            if let Ok(value) = serde_json::from_str::<Value>(data.trim()) {
                if value.get("id").is_some() && (value.get("result").is_some() || value.get("error").is_some()) {
                    last_response = Some(value);
                }
            }
        }
    }
    last_response
}

fn truncate(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sse_response() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
        let parsed = parse_sse_json_rpc(body).unwrap();
        assert_eq!(parsed["result"]["ok"], Value::Bool(true));
    }

    #[test]
    fn sse_without_response_is_none() {
        assert!(parse_sse_json_rpc("data: {\"jsonrpc\":\"2.0\",\"method\":\"x\"}\n\n").is_none());
    }
}
