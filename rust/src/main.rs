mod audio;
mod config;
mod error;
mod routes;
mod state;

use axum::routing::{get, post};
use axum::Router;
use std::net::SocketAddr;
use std::path::Path;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{fmt, EnvFilter};

use config::ServerConfig;
use state::{new_app_state, Models};

use qwen3_asr::inference::AsrInference;
use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use qwen3_tts::tensor::Device as TtsDevice;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Parse configuration from environment variables
    let config = ServerConfig::from_env().map_err(|e| anyhow::anyhow!("{}", e))?;

    // Determine TTS device based on backend
    #[cfg(feature = "tch-backend")]
    let tts_device = {
        if tch::Cuda::is_available() {
            tracing::info!("TTS using CUDA GPU");
            TtsDevice::Gpu(0)
        } else {
            tracing::info!("TTS using CPU");
            TtsDevice::Cpu
        }
    };

    #[cfg(feature = "mlx")]
    let tts_device = {
        qwen3_tts::backend::mlx::stream::init_mlx(true);
        tracing::info!("TTS using MLX Metal GPU");
        TtsDevice::Gpu(0)
    };

    // Load models
    let mut models = Models {
        custom_voice: None,
        base_model: None,
        speaker_encoder: None,
        audio_encoder: None,
        asr: None,
    };

    if let Some(ref path) = config.tts_customvoice_model_path {
        tracing::info!("Loading CustomVoice TTS model from {}", path);
        let inference = TTSInference::new(Path::new(path), tts_device)?;
        tracing::info!("CustomVoice TTS model loaded successfully");
        models.custom_voice = Some(inference);
    }

    if let Some(ref path) = config.tts_base_model_path {
        tracing::info!("Loading Base TTS model from {}", path);
        let inference = TTSInference::new(Path::new(path), tts_device)?;

        // Load speaker encoder from base model weights
        let se_config = inference.config().speaker_encoder_config.clone();
        let speaker_encoder =
            SpeakerEncoder::load(inference.weights(), &se_config, tts_device)?;
        tracing::info!("Speaker encoder loaded");

        // Load audio encoder for ICL voice cloning (if speech_tokenizer exists)
        let speech_tokenizer_path = Path::new(path)
            .join("speech_tokenizer")
            .join("model.safetensors");
        if speech_tokenizer_path.exists() {
            let audio_encoder = AudioEncoder::load(&speech_tokenizer_path, tts_device)?;
            tracing::info!("Audio encoder loaded for ICL mode");
            models.audio_encoder = Some(audio_encoder);
        } else {
            tracing::warn!(
                "speech_tokenizer not found at {}; ICL voice cloning will not be available",
                speech_tokenizer_path.display()
            );
        }

        models.speaker_encoder = Some(speaker_encoder);
        models.base_model = Some(inference);
        tracing::info!("Base TTS model loaded successfully");
    }

    if let Some(ref path) = config.asr_model_path {
        tracing::info!("Loading ASR model from {}", path);
        let model_dir = Path::new(path);

        #[cfg(feature = "tch-backend")]
        let asr_device = {
            if tch::Cuda::is_available() {
                tracing::info!("ASR using CUDA GPU");
                qwen3_asr::tensor::Device::Gpu(0)
            } else {
                tracing::info!("ASR using CPU");
                qwen3_asr::tensor::Device::Cpu
            }
        };

        #[cfg(feature = "mlx")]
        let asr_device = {
            // init_mlx is idempotent if already called for TTS
            qwen3_asr::backend::mlx::stream::init_mlx(true);
            tracing::info!("ASR using MLX Metal GPU");
            qwen3_asr::tensor::Device::Gpu(0)
        };

        let asr = AsrInference::load(model_dir, asr_device)?;
        tracing::info!("ASR model loaded successfully");
        models.asr = Some(asr);
    }

    let state = new_app_state(models);

    // Build router
    let app = Router::new()
        .route("/v1/audio/speech", post(routes::speech::speech_handler))
        .route(
            "/v1/audio/transcriptions",
            post(routes::transcription::transcribe),
        )
        .route("/v1/models", get(routes::models::list_models))
        .route("/health", get(routes::health::health))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid bind address");
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
