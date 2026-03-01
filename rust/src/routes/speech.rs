use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, State};
use axum::http::{header, Request};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::sync::{MutexGuard, TryLockError};
use std::time::Duration;

use crate::audio::{apply_speed, encode_audio};
use crate::config::{resolve_voice, ResponseFormat};
use crate::error::ApiError;
use crate::state::{AcquireSpeechSlotError, AppState, Models};

const LOCK_RETRY_INTERVAL_MS: u64 = 20;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default)]
    pub response_format: ResponseFormat,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_language")]
    pub language: String,
    pub instructions: Option<String>,
    /// Base64-encoded reference audio for voice cloning
    pub audio_sample: Option<String>,
    /// Transcript of reference audio (enables ICL mode)
    pub audio_sample_text: Option<String>,
}

fn default_voice() -> String {
    "alloy".to_string()
}
fn default_speed() -> f32 {
    1.0
}
fn default_language() -> String {
    "Auto".to_string()
}

/// Unified parameters parsed from either JSON or multipart.
struct SpeechParams {
    input: String,
    voice: String,
    response_format: ResponseFormat,
    speed: f32,
    language: String,
    instructions: Option<String>,
    audio_sample: Option<AudioSampleData>,
    audio_sample_text: Option<String>,
}

enum AudioSampleData {
    /// Base64-encoded audio string (from JSON body)
    Base64(String),
    /// Raw audio bytes (from multipart file upload)
    Bytes(Vec<u8>),
}

#[derive(Debug, Serialize)]
struct CancelSpeechResponse {
    ok: bool,
    message: String,
    current_task_id: Option<u64>,
    cancelled_up_to_task_id: u64,
}

struct CurrentSpeechTaskGuard {
    state: AppState,
    task_id: u64,
}

impl CurrentSpeechTaskGuard {
    fn new(state: AppState, task_id: u64) -> Self {
        state.speech_tasks.set_current_task_id(task_id);
        Self { state, task_id }
    }
}

impl Drop for CurrentSpeechTaskGuard {
    fn drop(&mut self) {
        self.state.speech_tasks.clear_current_task_id(self.task_id);
    }
}

// ---------------------------------------------------------------------------
// Handler: inspects Content-Type and dispatches
// ---------------------------------------------------------------------------

pub async fn speech_handler(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
) -> Result<Response, ApiError> {
    let _slot = state
        .acquire_speech_slot()
        .await
        .map_err(map_acquire_slot_error)?;

    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.contains("multipart/form-data") {
        let multipart = Multipart::from_request(request, &()).await.map_err(|e| {
            ApiError::unprocessable(format!("Failed to parse multipart request: {e}"))
        })?;
        speech_multipart(state.clone(), multipart).await
    } else {
        let bytes = Bytes::from_request(request, &()).await.map_err(
            |e: axum::extract::rejection::BytesRejection| {
                ApiError::unprocessable(format!("Failed to read request body: {e}"))
            },
        )?;
        if bytes.len() > state.speech_runtime.max_request_body_bytes {
            return Err(ApiError::payload_too_large(format!(
                "request body exceeds {} bytes",
                state.speech_runtime.max_request_body_bytes
            )));
        }
        let req: SpeechRequest = serde_json::from_slice(&bytes)
            .map_err(|e| ApiError::unprocessable(format!("Invalid JSON: {e}")))?;
        speech_json(state.clone(), req).await
    }
}

pub async fn cancel_current_handler(
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse {
    let current = state.speech_tasks.cancel_current();
    let current_after = state.speech_tasks.current_task_id();
    let cancelled_up_to_task_id = current.unwrap_or(0);
    let message = if let Some(task_id) = current {
        format!("Cancellation requested for current speech task {task_id}")
    } else {
        "No running speech task to cancel".to_string()
    };

    axum::Json(CancelSpeechResponse {
        ok: true,
        message,
        current_task_id: current_after,
        cancelled_up_to_task_id,
    })
}

pub async fn cancel_all_handler(
    State(state): State<AppState>,
) -> impl axum::response::IntoResponse {
    let cancelled_up_to_task_id = state.speech_tasks.cancel_all();
    let current_task_id = state.speech_tasks.current_task_id();

    axum::Json(CancelSpeechResponse {
        ok: true,
        message: format!(
            "Cancellation requested for all speech tasks up to {cancelled_up_to_task_id}"
        ),
        current_task_id,
        cancelled_up_to_task_id,
    })
}

async fn speech_json(state: AppState, req: SpeechRequest) -> Result<Response, ApiError> {
    validate_input(&req.input, req.speed)?;

    let params = SpeechParams {
        input: req.input,
        voice: req.voice,
        response_format: req.response_format,
        speed: req.speed,
        language: req.language,
        instructions: req.instructions,
        audio_sample: req.audio_sample.map(AudioSampleData::Base64),
        audio_sample_text: req.audio_sample_text,
    };

    generate_speech(state, params).await
}

async fn speech_multipart(state: AppState, mut multipart: Multipart) -> Result<Response, ApiError> {
    let mut input: Option<String> = None;
    let mut voice = "alloy".to_string();
    let mut response_format = ResponseFormat::Mp3;
    let mut speed: f32 = 1.0;
    let mut language = "Auto".to_string();
    let mut instructions: Option<String> = None;
    let mut audio_sample: Option<AudioSampleData> = None;
    let mut audio_sample_text: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::unprocessable(format!("Multipart parse error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "input" => {
                input = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            "model" => {
                let _ = field.text().await;
            }
            "voice" => {
                voice = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
            }
            "response_format" => {
                let fmt_str = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                response_format =
                    serde_json::from_value(serde_json::Value::String(fmt_str.clone())).map_err(
                        |_| ApiError::unprocessable(format!("Invalid response_format: {fmt_str}")),
                    )?;
            }
            "speed" => {
                let s = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                speed = s
                    .parse()
                    .map_err(|_| ApiError::unprocessable("Invalid speed value"))?;
            }
            "language" => {
                language = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
            }
            "instructions" => {
                instructions = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            "audio_sample" => {
                let ct = field.content_type().map(|s| s.to_string());
                if ct.is_some() {
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                    validate_audio_sample_size(&state, bytes.len())?;
                    audio_sample = Some(AudioSampleData::Bytes(bytes.to_vec()));
                } else {
                    let text = field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?;
                    audio_sample = Some(AudioSampleData::Base64(text));
                }
            }
            "audio_sample_text" => {
                audio_sample_text = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let input = input.ok_or_else(|| ApiError::unprocessable("'input' field is required"))?;
    validate_input(&input, speed)?;

    let params = SpeechParams {
        input,
        voice,
        response_format,
        speed,
        language,
        instructions,
        audio_sample,
        audio_sample_text,
    };

    generate_speech(state, params).await
}

// ---------------------------------------------------------------------------
// Core generation logic
// ---------------------------------------------------------------------------

async fn generate_speech(state: AppState, params: SpeechParams) -> Result<Response, ApiError> {
    let task_id = state.speech_tasks.alloc_task_id();
    let speed = params.speed;
    let format = params.response_format;
    let SpeechParams {
        input,
        voice,
        response_format: _,
        speed: _,
        language,
        instructions,
        audio_sample,
        audio_sample_text,
    } = params;
    let segment_max_chars = state.speech_runtime.segment_max_chars();
    let segments = split_text_for_tts(&input, state.speech_runtime.segment_max_bytes, segment_max_chars);
    tracing::info!(
        "speech task {} accepted: input_chars={}, segments={}, segment_max_bytes={}, segment_max_chars={}, segment_target_max_codes={}",
        task_id,
        input.chars().count(),
        segments.len(),
        state.speech_runtime.segment_max_bytes,
        segment_max_chars,
        state.speech_runtime.segment_target_max_codes
    );

    let (waveform, sample_rate) = tokio::task::spawn_blocking(move || {
        let models = lock_models_with_cancellation(&state, task_id)?;
        let _task_guard = CurrentSpeechTaskGuard::new(state.clone(), task_id);
        ensure_task_not_cancelled(&state, task_id)?;

        let mut effective_audio_sample = audio_sample;
        let mut effective_audio_sample_text = audio_sample_text;

        if effective_audio_sample.is_none() {
            if let Some(default_audio_bytes) = models.default_audio_sample_wav_bytes.as_ref() {
                effective_audio_sample = Some(AudioSampleData::Bytes(default_audio_bytes.clone()));
                if effective_audio_sample_text.is_none() {
                    effective_audio_sample_text = models.default_audio_sample_text.clone();
                }
            }
        }

        if let Some(audio_data) = effective_audio_sample {
            // Voice cloning path
            let base_model = models.base_model.as_ref().ok_or_else(|| {
                ApiError::bad_request(
                    "audio_sample requires a base model. \
                     Set TTS_BASE_MODEL_PATH to enable voice cloning.",
                )
            })?;
            let speaker_encoder = models.speaker_encoder.as_ref().ok_or_else(|| {
                ApiError::internal("Speaker encoder not loaded")
            })?;

            // Decode reference audio to f32 samples
            let (ref_samples, ref_sr) = match audio_data {
                AudioSampleData::Base64(b64) => {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&b64)
                        .map_err(|e| ApiError::bad_request(format!("Invalid base64: {e}")))?;
                    validate_audio_sample_size(&state, bytes.len())?;
                    qwen3_tts::audio::load_wav_bytes(&bytes)
                        .map_err(|e| ApiError::bad_request(format!("Invalid audio: {e}")))?
                }
                AudioSampleData::Bytes(bytes) => {
                    validate_audio_sample_size(&state, bytes.len())?;
                    qwen3_tts::audio::load_wav_bytes(&bytes)
                        .map_err(|e| ApiError::bad_request(format!("Invalid audio: {e}")))?
                }
            };

            // Resample to 24kHz for speaker encoder
            let ref_samples_24k = if ref_sr != 24000 {
                qwen3_tts::audio::resample(&ref_samples, ref_sr, 24000)
                    .map_err(|e| ApiError::internal(e.to_string()))?
            } else {
                ref_samples
            };

            // Extract speaker embedding
            let speaker_embedding = speaker_encoder
                .extract_embedding(&ref_samples_24k)
                .map_err(|e| ApiError::internal(e.to_string()))?;

            let mut merged_waveform = Vec::new();
            let mut merged_sample_rate = None;

            if let Some(icl_text) = effective_audio_sample_text.as_deref() {
                // ICL mode: encode reference audio to codec tokens
                let audio_encoder = models.audio_encoder.as_ref().ok_or_else(|| {
                    ApiError::internal(
                        "Audio encoder not loaded for ICL mode. \
                         Ensure speech_tokenizer/model.safetensors exists in the base model directory.",
                    )
                })?;
                let ref_codes = audio_encoder
                    .encode(&ref_samples_24k)
                    .map_err(|e| ApiError::internal(e.to_string()))?;

                for (segment_idx, segment) in segments.iter().enumerate() {
                    ensure_task_not_cancelled(&state, task_id)?;
                    let max_codes = state.speech_runtime.estimate_segment_max_codes(segment);
                    tracing::info!(
                        "speech task {} segment {}/{}: chars={}, max_codes={}",
                        task_id,
                        segment_idx + 1,
                        segments.len(),
                        segment.chars().count(),
                        max_codes
                    );
                    let (segment_waveform, segment_sample_rate) = base_model
                        .generate_with_icl(
                            segment,
                            icl_text,
                            &ref_codes,
                            &speaker_embedding,
                            &language,
                            0.9,
                            50,
                            max_codes,
                        )
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                    append_segment_audio(
                        &mut merged_waveform,
                        &mut merged_sample_rate,
                        segment_waveform,
                        segment_sample_rate,
                        state.speech_runtime.segment_gap_seconds,
                    )?;
                }
            } else {
                // X-vector only mode
                for (segment_idx, segment) in segments.iter().enumerate() {
                    ensure_task_not_cancelled(&state, task_id)?;
                    let max_codes = state.speech_runtime.estimate_segment_max_codes(segment);
                    tracing::info!(
                        "speech task {} segment {}/{}: chars={}, max_codes={}",
                        task_id,
                        segment_idx + 1,
                        segments.len(),
                        segment.chars().count(),
                        max_codes
                    );
                    let (segment_waveform, segment_sample_rate) = base_model
                        .generate_with_xvector(
                            segment,
                            &speaker_embedding,
                            &language,
                            0.9,
                            50,
                            max_codes,
                        )
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                    append_segment_audio(
                        &mut merged_waveform,
                        &mut merged_sample_rate,
                        segment_waveform,
                        segment_sample_rate,
                        state.speech_runtime.segment_gap_seconds,
                    )?;
                }
            }
            Ok::<(Vec<f32>, u32), ApiError>((merged_waveform, merged_sample_rate.unwrap_or(24000)))
        } else {
            // CustomVoice path
            let model = models.custom_voice.as_ref().ok_or_else(|| {
                ApiError::bad_request(
                    "Custom-voice model is not loaded. \
                     Set TTS_CUSTOMVOICE_MODEL_PATH to enable speaker voices, \
                     or provide audio_sample for voice cloning.",
                )
            })?;

            let speaker_name = resolve_voice(&voice).map_err(ApiError::bad_request)?;

            let instruct = instructions
                .as_deref()
                .or(models.default_instructions.as_deref())
                .unwrap_or("");

            let mut merged_waveform = Vec::new();
            let mut merged_sample_rate = None;
            for (segment_idx, segment) in segments.iter().enumerate() {
                ensure_task_not_cancelled(&state, task_id)?;
                let max_codes = state.speech_runtime.estimate_segment_max_codes(segment);
                tracing::info!(
                    "speech task {} segment {}/{}: chars={}, max_codes={}",
                    task_id,
                    segment_idx + 1,
                    segments.len(),
                    segment.chars().count(),
                    max_codes
                );
                let (segment_waveform, segment_sample_rate) = model
                    .generate_with_instruct(
                        segment,
                        &speaker_name,
                        &language,
                        instruct,
                        0.9,
                        50,
                        max_codes,
                    )
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                append_segment_audio(
                    &mut merged_waveform,
                    &mut merged_sample_rate,
                    segment_waveform,
                    segment_sample_rate,
                    state.speech_runtime.segment_gap_seconds,
                )?;
            }
            Ok::<(Vec<f32>, u32), ApiError>((merged_waveform, merged_sample_rate.unwrap_or(24000)))
        }
    })
    .await
    .map_err(|e| ApiError::internal(format!("Task join error: {e}")))??;

    // Apply speed adjustment
    let waveform = if (speed - 1.0).abs() > f32::EPSILON {
        apply_speed(&waveform, speed)
    } else {
        waveform
    };

    // Encode to requested format
    let audio_bytes = encode_audio(&waveform, sample_rate, format)?;

    Ok(([(header::CONTENT_TYPE, format.content_type())], audio_bytes).into_response())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_input(input: &str, speed: f32) -> Result<(), ApiError> {
    if input.trim().is_empty() {
        return Err(ApiError::unprocessable("'input' field is required"));
    }
    if !(0.25..=4.0).contains(&speed) {
        return Err(ApiError::unprocessable(
            "'speed' must be between 0.25 and 4.0",
        ));
    }
    Ok(())
}

fn validate_audio_sample_size(state: &AppState, bytes: usize) -> Result<(), ApiError> {
    if bytes > state.speech_runtime.max_audio_sample_bytes {
        return Err(ApiError::payload_too_large(format!(
            "audio_sample exceeds {} bytes",
            state.speech_runtime.max_audio_sample_bytes
        )));
    }
    Ok(())
}

fn map_acquire_slot_error(err: AcquireSpeechSlotError) -> ApiError {
    match err {
        AcquireSpeechSlotError::QueueFull { queued, limit } => ApiError::too_many_requests(
            format!(
                "speech queue is full (queued={queued}, limit={limit}); retry later or reduce concurrency"
            ),
        ),
        AcquireSpeechSlotError::GateClosed => {
            ApiError::internal("speech request gate is unavailable")
        }
    }
}

fn lock_models_with_cancellation<'a>(
    state: &'a AppState,
    task_id: u64,
) -> Result<MutexGuard<'a, Models>, ApiError> {
    loop {
        ensure_task_not_cancelled(state, task_id)?;
        match state.models.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::WouldBlock) => {
                std::thread::sleep(Duration::from_millis(LOCK_RETRY_INTERVAL_MS));
            }
            Err(TryLockError::Poisoned(e)) => {
                tracing::warn!("recovering from poisoned models mutex in speech route");
                state.models.clear_poison();
                return Ok(e.into_inner());
            }
        }
    }
}

fn ensure_task_not_cancelled(state: &AppState, task_id: u64) -> Result<(), ApiError> {
    if state.speech_tasks.is_cancelled(task_id) {
        return Err(ApiError::bad_request(format!(
            "speech task {task_id} was cancelled"
        )));
    }
    Ok(())
}

fn append_segment_audio(
    merged_waveform: &mut Vec<f32>,
    merged_sample_rate: &mut Option<u32>,
    segment_waveform: Vec<f32>,
    segment_sample_rate: u32,
    segment_gap_seconds: f32,
) -> Result<(), ApiError> {
    if let Some(rate) = *merged_sample_rate {
        if rate != segment_sample_rate {
            return Err(ApiError::internal(format!(
                "inconsistent segment sample rates: {rate} vs {segment_sample_rate}"
            )));
        }
    } else {
        *merged_sample_rate = Some(segment_sample_rate);
    }

    if !merged_waveform.is_empty() {
        let gap_samples = ((segment_sample_rate as f32) * segment_gap_seconds).round() as usize;
        merged_waveform.extend(std::iter::repeat_n(0.0, gap_samples));
    }
    merged_waveform.extend(segment_waveform);
    Ok(())
}

fn split_text_for_tts(input: &str, max_bytes: usize, max_chars: usize) -> Vec<String> {
    let text = input.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let normalized = text.replace("\r\n", "\n");
    let mut chunks = Vec::new();
    for paragraph in normalized
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        split_block_for_tts(paragraph, max_bytes, max_chars, &mut chunks);
    }

    if chunks.is_empty() {
        vec![normalized]
    } else {
        chunks
    }
}

fn split_block_for_tts(block: &str, max_bytes: usize, max_chars: usize, out: &mut Vec<String>) {
    let mut start = 0usize;
    while start < block.len() {
        start = skip_whitespace(block, start);
        if start >= block.len() {
            return;
        }

        let hard_limit = advance_within_budget(block, start, max_bytes, max_chars);
        let mut split = if hard_limit >= block.len() {
            block.len()
        } else {
            find_last_preferred_boundary(block, start, hard_limit).unwrap_or(hard_limit)
        };

        if split <= start {
            if let Some(ch) = block[start..].chars().next() {
                split = start + ch.len_utf8();
            } else {
                break;
            }
        }

        let chunk = block[start..split].trim();
        if !chunk.is_empty() {
            out.push(chunk.to_string());
        }
        start = split;
    }
}

fn skip_whitespace(text: &str, mut idx: usize) -> usize {
    while idx < text.len() {
        let Some(ch) = text[idx..].chars().next() else {
            break;
        };
        if ch.is_whitespace() {
            idx += ch.len_utf8();
            continue;
        }
        break;
    }
    idx
}

fn advance_within_budget(text: &str, start: usize, max_bytes: usize, max_chars: usize) -> usize {
    let mut bytes_used = 0usize;
    let mut end = start;
    for (chars_used, (off, ch)) in text[start..].char_indices().enumerate() {
        let ch_bytes = ch.len_utf8();
        if bytes_used + ch_bytes > max_bytes || chars_used + 1 > max_chars {
            break;
        }
        bytes_used += ch_bytes;
        end = start + off + ch_bytes;
    }

    if end > start {
        end
    } else if let Some(ch) = text[start..].chars().next() {
        start + ch.len_utf8()
    } else {
        start
    }
}

fn is_preferred_boundary(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r'
            | '。'
            | '！'
            | '？'
            | '；'
            | '.'
            | '!'
            | '?'
            | ';'
    )
}

fn find_last_preferred_boundary(text: &str, start: usize, hard_limit: usize) -> Option<usize> {
    let mut last = None;
    for (off, ch) in text[start..hard_limit].char_indices() {
        if is_preferred_boundary(ch) {
            last = Some(start + off + ch.len_utf8());
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::split_text_for_tts;

    const TEST_MAX_BYTES: usize = 4096;
    const TEST_MAX_CHARS: usize = 300;

    #[test]
    fn split_text_keeps_short_input_single_segment() {
        let input = "这是短文本。";
        let chunks = split_text_for_tts(input, TEST_MAX_BYTES, TEST_MAX_CHARS);
        assert_eq!(chunks, vec![input.to_string()]);
    }

    #[test]
    fn split_text_breaks_long_chinese_input() {
        let input = "第一段。第二段！第三段？".repeat(600);
        let chunks = split_text_for_tts(&input, TEST_MAX_BYTES, TEST_MAX_CHARS);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
        assert!(chunks.iter().all(|chunk| chunk.len() <= TEST_MAX_BYTES));
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= TEST_MAX_CHARS));
    }

    #[test]
    fn split_text_handles_long_unpunctuated_input() {
        let input = "啊".repeat(5000);
        let chunks = split_text_for_tts(&input, TEST_MAX_BYTES, TEST_MAX_CHARS);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.len() <= TEST_MAX_BYTES));
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= TEST_MAX_CHARS));
    }

    #[test]
    fn split_text_prefers_sentence_boundaries_under_limit() {
        let input = "第一段。第二段。第三段。";
        let chunks = split_text_for_tts(input, 16, 16);
        assert_eq!(
            chunks,
            vec![
                "第一段。".to_string(),
                "第二段。".to_string(),
                "第三段。".to_string()
            ]
        );
    }

    #[test]
    fn split_text_prefers_paragraph_boundaries() {
        let input = "第一段第一句。\n\n第二段第一句。";
        let chunks = split_text_for_tts(input, TEST_MAX_BYTES, TEST_MAX_CHARS);
        assert_eq!(
            chunks,
            vec!["第一段第一句。".to_string(), "第二段第一句。".to_string()]
        );
    }
}
