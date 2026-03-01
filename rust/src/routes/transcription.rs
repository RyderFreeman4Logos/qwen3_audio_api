use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use std::io::Write;

use crate::audio::convert_audio_to_wav_bytes;
use crate::error::ApiError;
use crate::state::AppState;

#[derive(Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

pub async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::unprocessable(format!("Multipart parse error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                file_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?
                        .to_vec(),
                );
            }
            "model" | "prompt" | "temperature" => {
                // Accepted for compatibility, not used
                let _ = field.text().await;
            }
            "language" => {
                language = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| ApiError::unprocessable(e.to_string()))?,
                );
            }
            "response_format" => {
                response_format = field
                    .text()
                    .await
                    .map_err(|e| ApiError::unprocessable(e.to_string()))?;
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let file_bytes =
        file_bytes.ok_or_else(|| ApiError::unprocessable("'file' field is required"))?;
    if file_bytes.is_empty() {
        return Err(ApiError::unprocessable("Empty audio file"));
    }

    // Determine file suffix from upload filename
    let suffix = file_name
        .as_deref()
        .and_then(|n| n.rsplit_once('.'))
        .map(|(_, ext)| format!(".{ext}"))
        .unwrap_or_else(|| ".wav".to_string());

    // Convert audio to WAV (mono, 16kHz, s16) using the statically-linked ffmpeg library
    let wav_bytes = convert_audio_to_wav_bytes(&file_bytes, &suffix, 16000)?;

    // Write WAV to a temp file for ASR inference
    let mut wav_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| ApiError::internal(format!("Failed to create WAV temp file: {e}")))?;
    wav_file
        .write_all(&wav_bytes)
        .map_err(|e| ApiError::internal(format!("Failed to write WAV temp file: {e}")))?;
    let wav_path = wav_file.path().to_string_lossy().to_string();

    // Run inference in blocking task
    let lang = language.clone();
    let wav_path_str = wav_path.clone();
    let text = tokio::task::spawn_blocking(move || {
        let models = state.lock_models_recover();
        let asr = models.asr.as_ref().ok_or_else(|| {
            ApiError::bad_request(
                "ASR model is not loaded. Set ASR_MODEL_PATH to enable transcription.",
            )
        })?;

        let result = asr
            .transcribe(&wav_path_str, lang.as_deref())
            .map_err(|e| ApiError::internal(e.to_string()))?;

        // Strip special token prefixes like <asr_text> from the output
        let text = result.text.trim().to_string();
        let text = text
            .strip_prefix("<asr_text>")
            .unwrap_or(&text)
            .trim()
            .to_string();
        Ok::<String, ApiError>(text)
    })
    .await
    .map_err(|e| ApiError::internal(format!("Task join error: {e}")))??;

    if response_format == "text" {
        Ok(([(header::CONTENT_TYPE, "text/plain")], text).into_response())
    } else {
        Ok(Json(TranscriptionResponse { text }).into_response())
    }
}
