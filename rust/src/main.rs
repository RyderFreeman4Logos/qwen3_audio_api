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

fn precompute_default_voice_clone_artifacts(models: &mut Models) -> anyhow::Result<()> {
    let Some(wav_bytes) = models.default_audio_sample_wav_bytes.as_ref() else {
        return Ok(());
    };
    let Some(speaker_encoder) = models.speaker_encoder.as_ref() else {
        tracing::debug!(
            "Skipping default voice-clone precompute: speaker encoder is unavailable (base model not loaded)"
        );
        return Ok(());
    };

    let (ref_samples, ref_sr) = qwen3_tts::audio::load_wav_bytes(wav_bytes)
        .map_err(|e| anyhow::anyhow!("failed to decode default reference WAV: {e}"))?;
    let ref_samples_24k = if ref_sr != 24000 {
        qwen3_tts::audio::resample(&ref_samples, ref_sr, 24000)
            .map_err(|e| anyhow::anyhow!("failed to resample default reference audio: {e}"))?
    } else {
        ref_samples
    };

    let speaker_embedding = speaker_encoder
        .extract_embedding(&ref_samples_24k)
        .map_err(|e| anyhow::anyhow!("failed to extract default speaker embedding: {e}"))?;
    models.default_audio_sample_speaker_embedding = Some(speaker_embedding);
    tracing::info!("Precomputed default speaker embedding for voice cloning");

    if models.default_audio_sample_text.is_some() {
        if let Some(audio_encoder) = models.audio_encoder.as_ref() {
            let ref_codes = audio_encoder.encode(&ref_samples_24k).map_err(|e| {
                anyhow::anyhow!("failed to encode default ICL reference codes: {e}")
            })?;
            tracing::info!(
                "Precomputed default ICL reference codes: {} frames",
                ref_codes.len()
            );
            models.default_audio_sample_ref_codes = Some(ref_codes);
        } else {
            tracing::warn!(
                "DEFAULT_AUDIO_SAMPLE_TEXT is set, but audio encoder is unavailable; \
                 default voice cloning will fall back to x-vector mode"
            );
            models.default_audio_sample_ref_codes = None;
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (silently ignored if missing)
    dotenvy::dotenv().ok();

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
        match config.tts_device.as_str() {
            "cpu" => {
                tracing::info!("TTS using CPU (TTS_DEVICE=cpu)");
                TtsDevice::Cpu
            }
            "cuda" | "gpu" => {
                if !tch::Cuda::is_available() {
                    tracing::warn!(
                        "TTS_DEVICE={} but tch reports CUDA unavailable; still forcing GPU(0)",
                        config.tts_device
                    );
                } else {
                    tracing::info!("TTS using CUDA GPU (TTS_DEVICE={})", config.tts_device);
                }
                TtsDevice::Gpu(0)
            }
            _ => {
                if tch::Cuda::is_available() {
                    tracing::info!("TTS using CUDA GPU (TTS_DEVICE=auto)");
                    TtsDevice::Gpu(0)
                } else {
                    tracing::info!("TTS using CPU (TTS_DEVICE=auto)");
                    TtsDevice::Cpu
                }
            }
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
        default_audio_sample_wav_bytes: None,
        default_audio_sample_text: config.default_audio_sample_text.clone(),
        default_audio_sample_speaker_embedding: None,
        default_audio_sample_ref_codes: None,
        default_instructions: config.default_instructions.clone(),
    };

    if let Some(ref path) = config.default_audio_sample_path {
        tracing::info!("Loading default reference audio from {}", path);
        let audio_bytes =
            std::fs::read(path).map_err(|e| anyhow::anyhow!("Failed to read {}: {}", path, e))?;
        let suffix = Path::new(path)
            .extension()
            .map(|ext| format!(".{}", ext.to_string_lossy()))
            .unwrap_or_else(|| ".wav".to_string());

        let wav_bytes = if suffix.eq_ignore_ascii_case(".wav") {
            audio_bytes
        } else {
            audio::convert_audio_to_wav_bytes(&audio_bytes, &suffix, 24000)
                .map_err(|e| anyhow::anyhow!("Failed to convert default audio to WAV: {}", e))?
        };

        models.default_audio_sample_wav_bytes = Some(wav_bytes);

        if models.default_audio_sample_text.is_some() {
            tracing::info!("Default voice cloning will run in ICL mode");
        } else {
            tracing::warn!(
                "DEFAULT_AUDIO_SAMPLE_TEXT is not set; default voice cloning will use x-vector mode"
            );
        }
    }

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
        let speaker_encoder = SpeakerEncoder::load(inference.weights(), &se_config, tts_device)?;
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

    if models.default_audio_sample_wav_bytes.is_some() {
        if let Err(err) = precompute_default_voice_clone_artifacts(&mut models) {
            tracing::warn!(
                "Failed to precompute default voice cloning artifacts; \
                 requests will compute on-demand: {}",
                err
            );
        }
    }

    if let Some(ref path) = config.asr_model_path {
        tracing::info!("Loading ASR model from {}", path);
        let model_dir = Path::new(path);

        #[cfg(feature = "tch-backend")]
        let asr_device = {
            match config.asr_device.as_str() {
                "cpu" => {
                    tracing::info!("ASR using CPU (ASR_DEVICE=cpu)");
                    qwen3_asr::tensor::Device::Cpu
                }
                "cuda" | "gpu" => {
                    if !tch::Cuda::is_available() {
                        tracing::warn!(
                            "ASR_DEVICE={} but tch reports CUDA unavailable; still forcing GPU(0)",
                            config.asr_device
                        );
                    } else {
                        tracing::info!("ASR using CUDA GPU (ASR_DEVICE={})", config.asr_device);
                    }
                    qwen3_asr::tensor::Device::Gpu(0)
                }
                _ => {
                    if tch::Cuda::is_available() {
                        tracing::info!("ASR using CUDA GPU (ASR_DEVICE=auto)");
                        qwen3_asr::tensor::Device::Gpu(0)
                    } else {
                        tracing::info!("ASR using CPU (ASR_DEVICE=auto)");
                        qwen3_asr::tensor::Device::Cpu
                    }
                }
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

    tracing::info!(
        "Speech runtime: segment_max_bytes={}, concurrent={}, queued={}, incremental={}, max_codes={}, vocoder_chunk={}, temperature={}, top_k={}",
        config.speech_runtime.segment_max_bytes,
        config.speech_runtime.max_concurrent_speech_requests,
        config.speech_runtime.max_queued_speech_requests,
        config.speech_runtime.incremental_enabled,
        config.speech_runtime.max_generation_codes,
        config.speech_runtime.vocoder_chunk_codes,
        config.speech_runtime.generation_temperature,
        config.speech_runtime.generation_top_k
    );

    let state = new_app_state(models, config.speech_runtime.clone());

    // Build router
    let app = Router::new()
        .route("/v1/audio/speech", post(routes::speech::speech_handler))
        .route(
            "/v1/audio/speech/cancel-current",
            post(routes::speech::cancel_current_handler),
        )
        .route(
            "/v1/audio/speech/cancel-all",
            post(routes::speech::cancel_all_handler),
        )
        .route(
            "/v1/audio/speech/status",
            get(routes::speech::status_handler),
        )
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
