use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
struct ModelInfo {
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelInfo>,
}

pub async fn list_models(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    let models = state.lock_models_recover();
    let mut data = Vec::new();

    if models.custom_voice.is_some() || models.base_model.is_some() {
        data.push(ModelInfo {
            id: "qwen3-tts",
            object: "model",
            owned_by: "qwen",
        });
    }

    if models.asr.is_some() {
        data.push(ModelInfo {
            id: "qwen3-asr",
            object: "model",
            owned_by: "qwen",
        });
    }

    Json(ModelsResponse {
        object: "list",
        data,
    })
}
