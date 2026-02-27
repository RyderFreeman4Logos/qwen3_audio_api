use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

pub async fn health() -> impl axum::response::IntoResponse {
    Json(HealthResponse { status: "ok" })
}
