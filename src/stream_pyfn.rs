//! The Python-facing entry point for `inference_stream`: a `#[pyfunction]`
//! (not a `polars_expr` plugin) because it needs to receive a real Python
//! callable (`on_token`) as a `PyObject` -- kwargs routed through the
//! `polars_expr` plugin boundary are serde-deserialized and cannot carry a
//! Python callback.
//!
//! See `src/model_client/streaming.rs` for the SSE parsing / provider fan-out
//! machinery this drives.

use crate::model_client::{self, streaming, Message, Provider, StreamEvent};
use crate::utils::RT;
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;
use std::str::FromStr;
use tokio::task::JoinHandle;

fn get_default_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => model_client::openai::DEFAULT_OPENAI_MODEL,
        Provider::Anthropic => model_client::anthropic::DEFAULT_ANTHROPIC_MODEL,
        Provider::Gemini => model_client::gemini::DEFAULT_GEMINI_MODEL,
        Provider::Groq => model_client::groq::DEFAULT_GROQ_MODEL,
        Provider::Bedrock => model_client::bedrock::DEFAULT_BEDROCK_MODEL,
    }
}

/// Per-row output: parallel `(text, finished)` vectors, one entry per input
/// row (`None` for a null input row in both).
type StreamBatchResult = (Vec<Option<String>>, Vec<Option<bool>>);

/// Resolve the provider/model pair exactly like
/// `expressions::resolve_provider_and_model`, but from raw `Option<&str>`
/// arguments instead of a deserialized kwargs struct (this entry point takes
/// its arguments directly as pyfunction parameters, not via the plugin/serde
/// boundary).
fn resolve_provider_and_model(provider: Option<&str>, model: Option<&str>) -> (Provider, String) {
    let provider = provider
        .and_then(|s| Provider::from_str(s).ok())
        .unwrap_or(Provider::OpenAI);
    let model = model
        .map(|s| s.to_string())
        .unwrap_or_else(|| get_default_model(provider).to_string());
    (provider, model)
}

/// Batch entry point for `inference_stream`. Returns `(text, finished)`
/// vectors, one entry per input row (`None`/`None` for a null input row).
///
/// See `polar_llama/__init__.py::inference_stream` for the Python-facing
/// wrapper and `docs/design` / the module docstring there for the full
/// streaming contract (struct dtype, cancellation semantics, etc).
#[pyfunction]
#[pyo3(signature = (inputs, provider=None, model=None, callback=None, messages=false))]
pub fn _stream_inference_batch(
    py: Python<'_>,
    inputs: Vec<Option<String>>,
    provider: Option<&str>,
    model: Option<&str>,
    callback: Option<Py<PyAny>>,
    messages: bool,
) -> PyResult<StreamBatchResult> {
    let (provider, model) = resolve_provider_and_model(provider, model);
    let n_rows = inputs.len();

    // Build (orig_idx, Vec<Message>) pairs for non-null rows. `messages=true`
    // rows are JSON message arrays parsed via parse_message_json; a parse
    // failure yields an immediately-resolved row (empty text, unfinished)
    // rather than failing the whole batch.
    let mut active: Vec<(usize, Vec<Message>)> = Vec::new();
    // Rows resolved before any network call happens at all (null input, or a
    // messages-parse failure): (orig_idx, text, finished).
    let mut precomputed: Vec<(usize, String, bool)> = Vec::new();
    let mut parse_notice: Option<String> = None;

    for (idx, opt_value) in inputs.into_iter().enumerate() {
        match opt_value {
            None => {
                // Null input row -> null struct row; tracked implicitly by
                // being absent from both `active` and `precomputed` and
                // defaulting to None at scatter time.
            }
            Some(value) => {
                if messages {
                    match crate::utils::parse_message_json(&value) {
                        Ok(msgs) => active.push((idx, msgs)),
                        Err(e) => {
                            precomputed.push((idx, String::new(), false));
                            parse_notice.get_or_insert_with(|| {
                                format!("row {idx}: failed to parse message JSON: {e}")
                            });
                        }
                    }
                } else {
                    active.push((
                        idx,
                        vec![Message {
                            role: "user".to_string(),
                            content: value,
                            cache_control: None,
                        }],
                    ));
                }
            }
        }
    }

    // Map original row index -> position in the active-rows accumulator
    // arrays, since a stream event's `row` field is the original index.
    let n_active = active.len();
    let mut orig_idx_of: Vec<usize> = Vec::with_capacity(n_active);
    for (orig_idx, _) in &active {
        orig_idx_of.push(*orig_idx);
    }
    let mut pos_of_orig: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(n_active);
    for (pos, orig_idx) in orig_idx_of.iter().enumerate() {
        pos_of_orig.insert(*orig_idx, pos);
    }

    let mut acc: Vec<String> = vec![String::new(); n_active];
    let mut finished: Vec<bool> = vec![false; n_active];
    let mut notice: Option<String> = parse_notice;

    if n_active > 0 {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, StreamEvent)>(256);
        let handle: JoinHandle<()> = RT.spawn(streaming::stream_batch(provider, model, active, tx));

        // Pump loop on the calling (Python/Polars) thread. This is the only
        // blocking point, and it runs ONLY inside `py.detach` (the current,
        // non-deprecated name for what used to be `allow_threads`): a
        // Python-side stall can therefore never wedge a tokio worker that is
        // blocked on sending into the bounded channel while this thread holds
        // the GIL, because the GIL is not held during the wait. The callback
        // below runs only after the GIL is reacquired, i.e. only on this
        // (the calling) thread, never from a tokio worker.
        loop {
            // Wake at least every 100ms even when no event arrives, so a
            // pending signal (Ctrl-C) is serviced promptly during a silent
            // post-connect stall -- not only when the next delta happens to
            // land. `blocking_recv()` alone would park indefinitely with the
            // GIL released, leaving check_signals() unreachable until traffic
            // resumed. RT.block_on runs on this (the calling) thread, which is
            // never a tokio worker, so it cannot deadlock the runtime.
            let recv = py.detach(|| {
                RT.block_on(async {
                    tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await
                })
            });

            // Service Ctrl-C on every wake, whether or not an event arrived.
            if let Err(_sig) = py.check_signals() {
                // Ctrl-C (or another pending signal-raised exception). Per the
                // owner ruling this becomes a partial-with-notice result, not
                // a raised exception: consume it, don't propagate it.
                notice = Some("interrupted by user (KeyboardInterrupt)".to_string());
                handle.abort();
                break;
            }

            let (row, evt) = match recv {
                Err(_elapsed) => continue, // 100ms tick, no event: re-check signals
                Ok(None) => break,         // channel closed = batch done
                Ok(Some(item)) => item,
            };
            let Some(&pos) = pos_of_orig.get(&row) else {
                // Defensive: an event for a row we didn't schedule. Should be
                // unreachable, but never let it panic the pump.
                continue;
            };

            match evt {
                StreamEvent::Delta(delta) => {
                    acc[pos].push_str(&delta);
                    if let Some(cb) = &callback {
                        if let Err(e) = cb.call1(py, (row, delta.as_str())) {
                            notice = Some(format!("on_token callback raised: {e}"));
                            handle.abort();
                            break;
                        }
                    }
                }
                StreamEvent::Done => finished[pos] = true,
                StreamEvent::Error(msg) => {
                    // finished stays false for this row.
                    notice.get_or_insert_with(|| format!("row {row} stream error: {msg}"));
                }
            }
        }
        drop(rx); // after abort: any sender still alive fails silently (`let _ = send`)
    }

    if let Some(notice_msg) = notice {
        let message = format!(
            "inference_stream: {notice_msg}; unfinished rows returned with finished=false"
        );
        // CString::new can fail only on embedded NUL bytes; message is our
        // own formatted text so this is not expected, but degrade instead of
        // panicking if it ever happens.
        if let Ok(c_message) = std::ffi::CString::new(message) {
            let category = py.get_type::<PyRuntimeWarning>();
            let _ = PyErr::warn(py, category.as_any(), &c_message, 1);
        }
    }

    // Scatter results back to original indices; null inputs and rows never
    // scheduled (shouldn't happen beyond null/parse-failure) -> (None, None).
    let mut out_text: Vec<Option<String>> = vec![None; n_rows];
    let mut out_finished: Vec<Option<bool>> = vec![None; n_rows];

    for (pos, orig_idx) in orig_idx_of.into_iter().enumerate() {
        out_text[orig_idx] = Some(std::mem::take(&mut acc[pos]));
        out_finished[orig_idx] = Some(finished[pos]);
    }
    for (orig_idx, text, is_finished) in precomputed {
        out_text[orig_idx] = Some(text);
        out_finished[orig_idx] = Some(is_finished);
    }

    Ok((out_text, out_finished))
}
