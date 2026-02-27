use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

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

    // Convert audio to WAV via ffmpeg (mono, 16kHz, s16)
    let wav_path = convert_audio_to_wav(&file_bytes, &suffix)?;

    // Run inference in blocking task
    let lang = language.clone();
    let wav_path_str = wav_path.clone();
    let text = tokio::task::spawn_blocking(move || {
        let models = state
            .lock()
            .map_err(|e| ApiError::internal(e.to_string()))?;
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

    // Clean up temp files
    let _ = std::fs::remove_file(&wav_path);

    if response_format == "text" {
        Ok(([(header::CONTENT_TYPE, "text/plain")], text).into_response())
    } else {
        Ok(Json(TranscriptionResponse { text }).into_response())
    }
}

/// Convert audio bytes to a WAV temp file using ffmpeg.
/// Returns the path to the temporary WAV file.
fn convert_audio_to_wav(audio_bytes: &[u8], suffix: &str) -> Result<String, ApiError> {
    // Write source audio to a temp file with original suffix
    let mut src_file = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .map_err(|e| ApiError::internal(format!("Failed to create temp file: {e}")))?;
    src_file
        .write_all(audio_bytes)
        .map_err(|e| ApiError::internal(format!("Failed to write temp file: {e}")))?;
    let src_path = src_file.path().to_string_lossy().to_string();

    // Create output WAV temp file
    let wav_file = NamedTempFile::new()
        .map_err(|e| ApiError::internal(format!("Failed to create WAV temp file: {e}")))?;
    let wav_path = format!("{}.wav", wav_file.path().to_string_lossy());
    // We use a separate path so ffmpeg can write to it
    drop(wav_file);

    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            &wav_path,
        ])
        .output()
        .map_err(|e| ApiError::internal(format!("Failed to run ffmpeg: {e}")))?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&wav_path);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ApiError::internal(format!(
            "ffmpeg audio conversion failed: {stderr}"
        )));
    }

    Ok(wav_path)
}
