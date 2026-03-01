use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, State};
use axum::http::{header, Request};
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde::Deserialize;

use crate::audio::{apply_speed, encode_audio};
use crate::config::{resolve_voice, ResponseFormat};
use crate::error::ApiError;
use crate::state::AppState;

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

// ---------------------------------------------------------------------------
// Handler: inspects Content-Type and dispatches
// ---------------------------------------------------------------------------

pub async fn speech_handler(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
) -> Result<Response, ApiError> {
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
        speech_multipart(state, multipart).await
    } else {
        let bytes = Bytes::from_request(request, &()).await.map_err(
            |e: axum::extract::rejection::BytesRejection| {
                ApiError::unprocessable(format!("Failed to read request body: {e}"))
            },
        )?;
        let req: SpeechRequest = serde_json::from_slice(&bytes)
            .map_err(|e| ApiError::unprocessable(format!("Invalid JSON: {e}")))?;
        speech_json(state, req).await
    }
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
    let speed = params.speed;
    let format = params.response_format;

    let (waveform, sample_rate) = tokio::task::spawn_blocking(move || {
        let models = state.lock().map_err(|e| ApiError::internal(e.to_string()))?;

        let mut effective_audio_sample = params.audio_sample;
        let mut effective_audio_sample_text = params.audio_sample_text;

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
                    qwen3_tts::audio::load_wav_bytes(&bytes)
                        .map_err(|e| ApiError::bad_request(format!("Invalid audio: {e}")))?
                }
                AudioSampleData::Bytes(bytes) => {
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

            let use_icl = effective_audio_sample_text.is_some();

            if use_icl {
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

                base_model
                    .generate_with_icl(
                        &params.input,
                        effective_audio_sample_text.as_deref().unwrap(),
                        &ref_codes,
                        &speaker_embedding,
                        &params.language,
                        0.9,
                        50,
                        2048,
                    )
                    .map_err(|e| ApiError::internal(e.to_string()))
            } else {
                // X-vector only mode
                base_model
                    .generate_with_xvector(
                        &params.input,
                        &speaker_embedding,
                        &params.language,
                        0.9,
                        50,
                        2048,
                    )
                    .map_err(|e| ApiError::internal(e.to_string()))
            }
        } else {
            // CustomVoice path
            let model = models.custom_voice.as_ref().ok_or_else(|| {
                ApiError::bad_request(
                    "Custom-voice model is not loaded. \
                     Set TTS_CUSTOMVOICE_MODEL_PATH to enable speaker voices, \
                     or provide audio_sample for voice cloning.",
                )
            })?;

            let speaker_name =
                resolve_voice(&params.voice).map_err(ApiError::bad_request)?;

            let instruct = params
                .instructions
                .as_deref()
                .or(models.default_instructions.as_deref())
                .unwrap_or("");

            model
                .generate_with_instruct(
                    &params.input,
                    &speaker_name,
                    &params.language,
                    instruct,
                    0.9,
                    50,
                    2048,
                )
                .map_err(|e| ApiError::internal(e.to_string()))
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
    if input.is_empty() {
        return Err(ApiError::unprocessable("'input' field is required"));
    }
    if input.len() > 4096 {
        return Err(ApiError::unprocessable("'input' exceeds 4096 characters"));
    }
    if !(0.25..=4.0).contains(&speed) {
        return Err(ApiError::unprocessable(
            "'speed' must be between 0.25 and 4.0",
        ));
    }
    Ok(())
}
