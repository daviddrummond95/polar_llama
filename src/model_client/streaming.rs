//! Server-Sent Events (SSE) streaming support shared across providers.
//!
//! This module implements an incremental SSE parser, provider-specific frame
//! decoders (OpenAI-compatible wire format and Anthropic's), and the batch
//! fan-out driver that turns a set of message arrays into a stream of
//! `(row, StreamEvent)` pairs delivered over a bounded mpsc channel.

use super::{Message, ModelClient, ModelClientError, Provider};
use futures::StreamExt;
use reqwest::Client;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

/// A single decoded streaming event for one row.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A text delta (partial token(s)) to append to the row's accumulated text.
    Delta(String),
    /// The stream reached a clean terminator ("[DONE]" / `message_stop`).
    Done,
    /// The stream failed or was cut off; `finished` stays false for this row.
    Error(String),
}

/// One complete SSE frame: an optional `event:` field name and the
/// concatenated `data:` payload (multi-line `data:` fields are joined with
/// `\n`, matching the SSE spec).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SseFrame {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE parser: buffers partial lines across network chunks and
/// yields complete (blank-line terminated) frames.
pub struct SseParser {
    buf: String,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Feed raw bytes (possibly a partial line, possibly several frames);
    /// returns every complete frame contained in the accumulated buffer.
    /// Handles LF and CRLF line endings, multi-line `data:` fields, `event:`
    /// fields, and `:`-prefixed comment lines (ignored).
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        // Lossy is fine here: SSE payloads are expected to be valid UTF-8;
        // any stray invalid bytes degrade gracefully instead of panicking.
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        // Normalize CRLF to LF so frame-boundary detection only has to look
        // for a single pattern ("\n\n"). Re-running this over the whole
        // (small, since we drain consumed frames below) buffer on every feed
        // is simpler and safer than trying to track a split "\r" across
        // chunk boundaries.
        if self.buf.contains('\r') {
            self.buf = self.buf.replace("\r\n", "\n");
        }

        let mut frames = Vec::new();
        // A frame ends at a blank line ("\n\n"). Process as many complete
        // frames as are currently buffered.
        while let Some(pos) = self.buf.find("\n\n") {
            let raw_frame = self.buf[..pos].to_string();
            self.buf.drain(..pos + 2);
            if let Some(frame) = parse_frame(&raw_frame) {
                frames.push(frame);
            }
        }

        frames
    }
}

/// Parse one raw frame (lines joined by `\n` or `\r\n`, no trailing blank
/// line) into an `SseFrame`. Lines starting with `:` are comments and
/// skipped. Unknown fields are ignored per the SSE spec.
fn parse_frame(raw: &str) -> Option<SseFrame> {
    let mut event: Option<String> = None;
    let mut data_lines: Vec<&str> = Vec::new();

    for line in raw.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix(':') {
            let _ = rest; // comment line, ignored
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim_start().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
        // Other fields (id:, retry:) are ignored; we don't need them.
    }

    if event.is_none() && data_lines.is_empty() {
        return None;
    }

    Some(SseFrame {
        event,
        data: data_lines.join("\n"),
    })
}

/// OpenAI-wire-format frame -> event. Shared by OpenAI and Groq (Groq is
/// OpenAI-SSE-compatible).
///
/// - `data: [DONE]` -> `Done`
/// - JSON with a top-level `error` object -> `Error`
/// - `choices[0].delta.content` present and non-empty -> `Delta`
/// - otherwise (e.g. a role-only delta, or an empty content chunk) -> `None`
///   (skip; not every frame carries text)
pub fn openai_frame_to_event(frame: &SseFrame) -> Option<StreamEvent> {
    let data = frame.data.trim();
    if data.is_empty() {
        return None;
    }
    if data == "[DONE]" {
        return Some(StreamEvent::Done);
    }

    let value: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return None,
    };

    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error")
            .to_string();
        return Some(StreamEvent::Error(message));
    }

    let content = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(|c| c.as_str());

    match content {
        Some(text) if !text.is_empty() => Some(StreamEvent::Delta(text.to_string())),
        _ => None,
    }
}

/// Anthropic frame -> event, keyed on the `event:` field (falling back to
/// `data.type` if `event:` is absent, which some proxies/mocks omit).
///
/// - `content_block_delta` with `delta.type == "text_delta"` -> `Delta(delta.text)`
/// - `message_stop` -> `Done`
/// - `error` -> `Error(error.message)`
/// - `ping` / `message_start` / `message_delta` / `content_block_start` /
///   `content_block_stop` -> `None` (ignored)
pub fn anthropic_frame_to_event(frame: &SseFrame) -> Option<StreamEvent> {
    let data = frame.data.trim();
    if data.is_empty() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let event_type = frame
        .event
        .clone()
        .or_else(|| value.get("type").and_then(|t| t.as_str()).map(String::from))?;

    match event_type.as_str() {
        "content_block_delta" => {
            let delta = value.get("delta")?;
            if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                let text = delta.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if text.is_empty() {
                    None
                } else {
                    Some(StreamEvent::Delta(text.to_string()))
                }
            } else {
                None
            }
        }
        "message_stop" => Some(StreamEvent::Done),
        "error" => {
            let message = value
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error")
                .to_string();
            Some(StreamEvent::Error(message))
        }
        "ping" | "message_start" | "message_delta" | "content_block_start"
        | "content_block_stop" => None,
        _ => None,
    }
}

/// Streaming-dedicated HTTP client: same builder as the default shared
/// client, but WITHOUT a total request timeout (reqwest's `.timeout()` covers
/// the whole body read, which would kill any stream longer than the limit).
/// `connect_timeout` bounds the initial connection; `read_timeout` bounds the
/// gap *between* body chunks, so a server that returns 200 headers and then
/// goes silent forever fails with a timeout instead of hanging the whole
/// `collect()` (which the pump could otherwise only escape via Ctrl-C). A
/// legitimately slow-but-progressing stream keeps resetting the per-read
/// clock, so a generous 120s window never truncates real output.
static STREAM_HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| Client::new())
});

/// Access the shared streaming HTTP client (no total timeout).
pub fn stream_http_client() -> &'static Client {
    &STREAM_HTTP_CLIENT
}

/// Shared driver for any OpenAI-SSE-compatible provider (OpenAI, Groq): sends
/// a streaming chat completion request and forwards decoded events to `tx`.
///
/// Frame emission stops after `Done`/`Error`. EOF without a terminator frame
/// is reported as an `Error`. A transport error while reading the body is
/// also reported as an `Error`. `tx.send` failures (receiver dropped, e.g.
/// after abort) are ignored -- the pump may have already stopped listening.
pub async fn run_openai_sse<T: ModelClient + Sync + ?Sized>(
    model_client: &T,
    http: &Client,
    messages: &[Message],
    row: usize,
    tx: &Sender<(usize, StreamEvent)>,
) -> Result<(), ModelClientError> {
    let mut body = model_client.format_request_body(messages, None, None);
    body["stream"] = serde_json::json!(true);

    let api_key = model_client.get_api_key();
    let request = model_client.apply_auth(http.post(model_client.api_endpoint()), &api_key);

    let response = request.json(&body).send().await?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(ModelClientError::Http(status.as_u16(), text));
    }

    drive_sse_stream(response, row, tx, openai_frame_to_event).await;
    Ok(())
}

/// Shared byte-stream -> frame -> event pump used by both provider drivers.
/// Stops forwarding events once `Done`/`Error` has been seen (but keeps
/// draining nothing further -- the function simply returns), and synthesizes
/// an `Error` if the stream ends (EOF or transport error) without ever
/// emitting a terminator.
pub(crate) async fn drive_sse_stream<F>(
    response: reqwest::Response,
    row: usize,
    tx: &Sender<(usize, StreamEvent)>,
    frame_to_event: F,
) where
    F: Fn(&SseFrame) -> Option<StreamEvent>,
{
    let mut parser = SseParser::new();
    let mut byte_stream = response.bytes_stream();
    let mut terminated = false;

    while let Some(item) = byte_stream.next().await {
        match item {
            Ok(bytes) => {
                for frame in parser.feed(&bytes) {
                    if let Some(event) = frame_to_event(&frame) {
                        let is_terminal = matches!(event, StreamEvent::Done | StreamEvent::Error(_));
                        let _ = tx.send((row, event)).await;
                        if is_terminal {
                            terminated = true;
                            break;
                        }
                    }
                }
                if terminated {
                    break;
                }
            }
            Err(e) => {
                let _ = tx.send((row, StreamEvent::Error(e.to_string()))).await;
                terminated = true;
                break;
            }
        }
    }

    if !terminated {
        let _ = tx
            .send((
                row,
                StreamEvent::Error("stream ended without completion marker".to_string()),
            ))
            .await;
    }
}

/// Batch fan-out mirroring `run_batch`: each row drives
/// `send_request_streaming` concurrently, bounded by `max_concurrency`. A
/// row-level `Err` becomes an `Error` event for that row. All `tx` clones
/// drop once every row's future completes, closing the channel -- this is
/// the pump's termination signal.
pub async fn stream_batch(
    provider: Provider,
    model: String,
    message_arrays: Vec<(usize, Vec<Message>)>,
    tx: Sender<(usize, StreamEvent)>,
) {
    let client = super::create_client(provider, &model);
    let http = stream_http_client().clone();

    let futures_iter = message_arrays.into_iter().map(|(row, messages)| {
        let client = &client;
        let http = &http;
        let tx = tx.clone();
        async move {
            if let Err(e) = client.send_request_streaming(http, &messages, row, &tx).await {
                let _ = tx.send((row, StreamEvent::Error(e.to_string()))).await;
            }
        }
    });

    futures::stream::iter(futures_iter)
        .buffered(super::max_concurrency())
        .collect::<Vec<()>>()
        .await;
    // `tx` (the original passed-in sender) and every per-row clone above are
    // dropped here, closing the channel so the pump's `recv()` returns `None`.
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // R1: SSE parser reassembles frames split across `feed()` calls.
    // ------------------------------------------------------------------
    #[test]
    fn sse_parser_reassembles_split_frames() {
        let mut parser = SseParser::new();
        let full = b"event: content_block_delta\ndata: {\"a\":1}\n\n";

        let mut frames = Vec::new();
        for byte in full {
            frames.extend(parser.feed(&[*byte]));
        }

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("content_block_delta"));
        assert_eq!(frames[0].data, "{\"a\":1}");

        // Chunk boundary mid `data:` line.
        let mut parser2 = SseParser::new();
        let mut frames2 = parser2.feed(b"data: {\"choi");
        assert!(frames2.is_empty());
        frames2.extend(parser2.feed(b"ces\":[]}\n\n"));
        assert_eq!(frames2.len(), 1);
        assert_eq!(frames2[0].data, "{\"choices\":[]}");
    }

    // ------------------------------------------------------------------
    // R2: CRLF line endings, multi-line data, and comment lines.
    // ------------------------------------------------------------------
    #[test]
    fn sse_parser_crlf_and_multiline_data_and_comments() {
        let mut parser = SseParser::new();
        let raw = b": this is a comment\r\nevent: message_stop\r\ndata: line1\r\ndata: line2\r\n\r\n";
        let frames = parser.feed(raw);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event.as_deref(), Some("message_stop"));
        assert_eq!(frames[0].data, "line1\nline2");

        // A pure comment-only frame yields no SseFrame at all (no event, no data).
        let mut parser2 = SseParser::new();
        let frames2 = parser2.feed(b": keep-alive\n\n");
        assert!(frames2.is_empty());
    }

    // ------------------------------------------------------------------
    // R3: OpenAI frame decoding: delta, [DONE], error, empty/absent delta.
    // ------------------------------------------------------------------
    #[test]
    fn openai_frame_to_event_delta_done_error() {
        let delta_frame = SseFrame {
            event: None,
            data: r#"{"choices":[{"delta":{"content":"Hello"}}]}"#.to_string(),
        };
        assert_eq!(
            openai_frame_to_event(&delta_frame),
            Some(StreamEvent::Delta("Hello".to_string()))
        );

        let done_frame = SseFrame {
            event: None,
            data: "[DONE]".to_string(),
        };
        assert_eq!(openai_frame_to_event(&done_frame), Some(StreamEvent::Done));

        let error_frame = SseFrame {
            event: None,
            data: r#"{"error":{"message":"rate limited"}}"#.to_string(),
        };
        assert_eq!(
            openai_frame_to_event(&error_frame),
            Some(StreamEvent::Error("rate limited".to_string()))
        );

        // Empty content -> skip.
        let empty_frame = SseFrame {
            event: None,
            data: r#"{"choices":[{"delta":{"content":""}}]}"#.to_string(),
        };
        assert_eq!(openai_frame_to_event(&empty_frame), None);

        // Absent delta content (e.g. role-only first chunk) -> skip.
        let role_only_frame = SseFrame {
            event: None,
            data: r#"{"choices":[{"delta":{"role":"assistant"}}]}"#.to_string(),
        };
        assert_eq!(openai_frame_to_event(&role_only_frame), None);
    }

    // ------------------------------------------------------------------
    // R4: Anthropic frame decoding: text_delta, message_stop, error, ignored types.
    // ------------------------------------------------------------------
    #[test]
    fn anthropic_frame_to_event_all_types() {
        let text_delta_frame = SseFrame {
            event: Some("content_block_delta".to_string()),
            data: r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}"#
                .to_string(),
        };
        assert_eq!(
            anthropic_frame_to_event(&text_delta_frame),
            Some(StreamEvent::Delta("Hi".to_string()))
        );

        let stop_frame = SseFrame {
            event: Some("message_stop".to_string()),
            data: r#"{"type":"message_stop"}"#.to_string(),
        };
        assert_eq!(anthropic_frame_to_event(&stop_frame), Some(StreamEvent::Done));

        let error_frame = SseFrame {
            event: Some("error".to_string()),
            data: r#"{"type":"error","error":{"message":"overloaded"}}"#.to_string(),
        };
        assert_eq!(
            anthropic_frame_to_event(&error_frame),
            Some(StreamEvent::Error("overloaded".to_string()))
        );

        for ignored in ["ping", "message_start", "message_delta", "content_block_start", "content_block_stop"] {
            let frame = SseFrame {
                event: Some(ignored.to_string()),
                data: format!(r#"{{"type":"{ignored}"}}"#),
            };
            assert_eq!(anthropic_frame_to_event(&frame), None, "expected {ignored} to be ignored");
        }
    }

    // ------------------------------------------------------------------
    // R5: default streaming fallback (trait default method) emits exactly
    // [Delta("hello"), Done] for a dummy ModelClient with zero network.
    // ------------------------------------------------------------------
    struct DummyClient;

    #[async_trait::async_trait]
    impl ModelClient for DummyClient {
        fn provider(&self) -> Provider {
            Provider::Gemini
        }
        fn api_endpoint(&self) -> String {
            "http://unused.invalid".to_string()
        }
        fn model_name(&self) -> &str {
            "dummy-model"
        }
        fn format_messages(&self, _messages: &[Message]) -> serde_json::Value {
            serde_json::json!([])
        }
        fn parse_response(&self, _response_text: &str) -> Result<String, ModelClientError> {
            Ok(String::new())
        }
        async fn send_request(
            &self,
            _client: &Client,
            _messages: &[Message],
        ) -> Result<String, ModelClientError> {
            Ok("hello".to_string())
        }
    }

    #[test]
    fn default_streaming_fallback_emits_delta_then_done() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, StreamEvent)>(8);
            let client = DummyClient;
            let http = Client::new();
            let messages = vec![Message {
                role: "user".to_string(),
                content: "hi".to_string(),
                cache_control: None,
            }];

            client
                .send_request_streaming(&http, &messages, 0, &tx)
                .await
                .unwrap();
            drop(tx);

            let mut events = Vec::new();
            while let Some(evt) = rx.recv().await {
                events.push(evt);
            }

            assert_eq!(
                events,
                vec![
                    (0, StreamEvent::Delta("hello".to_string())),
                    (0, StreamEvent::Done),
                ]
            );
        });
    }
}
