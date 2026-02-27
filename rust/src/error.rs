use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Unified error type for the API server.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn unprocessable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": serde_json::Value::Null,
            }
        });
        (self.status, axum::Json(body)).into_response()
    }
}

impl From<qwen3_tts::error::Qwen3TTSError> for ApiError {
    fn from(e: qwen3_tts::error::Qwen3TTSError) -> Self {
        ApiError::internal(e.to_string())
    }
}

impl From<qwen3_asr::error::AsrError> for ApiError {
    fn from(e: qwen3_asr::error::AsrError) -> Self {
        ApiError::internal(e.to_string())
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status, self.message)
    }
}
