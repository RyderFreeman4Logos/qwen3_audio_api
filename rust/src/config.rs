use std::collections::{HashMap, HashSet};

const DEFAULT_SPEECH_SEGMENT_MAX_BYTES: usize = 1024;
const DEFAULT_SEGMENT_GAP_SECONDS: f32 = 0.12;
const DEFAULT_MAX_CONCURRENT_SPEECH_REQUESTS: usize = 1;
const DEFAULT_MAX_QUEUED_SPEECH_REQUESTS: usize = 8;
const DEFAULT_MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_AUDIO_SAMPLE_BYTES: usize = 20 * 1024 * 1024;
const DEFAULT_LEGACY_FIXED_MAX_CODES: i64 = 2048;
const DEFAULT_MIN_GENERATION_CODES: i64 = 192;
const DEFAULT_MAX_GENERATION_CODES: i64 = 2048;
const DEFAULT_BASE_GENERATION_CODES: i64 = 96;
const DEFAULT_CODES_PER_CHAR: f32 = 3.8;
const DEFAULT_VOCODER_CHUNK_CODES: i64 = 0;
const DEFAULT_SEGMENT_TARGET_MAX_CODES: i64 = 640;
const DEFAULT_SEGMENT_BREAK_ON_COMMAS: bool = false;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

/// Runtime controls for speech synthesis safety and performance.
#[derive(Clone, Debug)]
pub struct SpeechRuntimeConfig {
    /// Maximum bytes per synthesized segment before automatic splitting.
    pub segment_max_bytes: usize,
    /// Gap inserted between merged segments.
    pub segment_gap_seconds: f32,
    /// Maximum concurrent speech requests executing inference.
    pub max_concurrent_speech_requests: usize,
    /// Maximum requests waiting for inference slot.
    pub max_queued_speech_requests: usize,
    /// Maximum JSON request body bytes.
    pub max_request_body_bytes: usize,
    /// Maximum decoded audio_sample bytes accepted from request.
    pub max_audio_sample_bytes: usize,
    /// Legacy fixed max code frames per segment when incremental mode is disabled.
    pub legacy_fixed_max_codes: i64,
    /// Adaptive generation lower bound for max codes per segment.
    pub min_generation_codes: i64,
    /// Adaptive generation upper bound for max codes per segment.
    pub max_generation_codes: i64,
    /// Adaptive max-codes base budget added to per-char estimate.
    pub base_generation_codes: i64,
    /// Adaptive max-codes slope per input character.
    pub codes_per_char: f32,
    /// Optional hard cap for generated codes per segment (0 = disabled).
    pub vocoder_chunk_codes: i64,
    /// Target upper budget used for text segmentation.
    /// Keeps each segment short enough to reduce long-context degeneration.
    pub segment_target_max_codes: i64,
    /// Enables comma-level split points for conservative long-text synthesis.
    pub segment_break_on_commas: bool,
    /// Enables adaptive generation budgeting (rollback: set to 0).
    pub incremental_enabled: bool,
}

impl SpeechRuntimeConfig {
    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self {
            segment_max_bytes: env_usize(
                "RUST_TTS_SEGMENT_MAX_BYTES",
                DEFAULT_SPEECH_SEGMENT_MAX_BYTES,
            ),
            segment_gap_seconds: env_f32(
                "RUST_TTS_SEGMENT_GAP_SECONDS",
                DEFAULT_SEGMENT_GAP_SECONDS,
            ),
            max_concurrent_speech_requests: env_usize(
                "RUST_TTS_MAX_CONCURRENT_REQUESTS",
                DEFAULT_MAX_CONCURRENT_SPEECH_REQUESTS,
            ),
            max_queued_speech_requests: env_usize(
                "RUST_TTS_MAX_QUEUED_REQUESTS",
                DEFAULT_MAX_QUEUED_SPEECH_REQUESTS,
            ),
            max_request_body_bytes: env_usize(
                "RUST_TTS_MAX_REQUEST_BODY_BYTES",
                DEFAULT_MAX_REQUEST_BODY_BYTES,
            ),
            max_audio_sample_bytes: env_usize(
                "RUST_TTS_MAX_AUDIO_SAMPLE_BYTES",
                DEFAULT_MAX_AUDIO_SAMPLE_BYTES,
            ),
            legacy_fixed_max_codes: env_i64(
                "RUST_TTS_LEGACY_MAX_CODES",
                DEFAULT_LEGACY_FIXED_MAX_CODES,
            ),
            min_generation_codes: env_i64(
                "RUST_TTS_MIN_GENERATION_CODES",
                DEFAULT_MIN_GENERATION_CODES,
            ),
            max_generation_codes: env_i64(
                "RUST_TTS_MAX_GENERATION_CODES",
                DEFAULT_MAX_GENERATION_CODES,
            ),
            base_generation_codes: env_i64(
                "RUST_TTS_BASE_GENERATION_CODES",
                DEFAULT_BASE_GENERATION_CODES,
            ),
            codes_per_char: env_f32("RUST_TTS_CODES_PER_CHAR", DEFAULT_CODES_PER_CHAR),
            vocoder_chunk_codes: env_i64("RUST_TTS_VOCODER_CHUNK", DEFAULT_VOCODER_CHUNK_CODES),
            segment_target_max_codes: env_i64(
                "RUST_TTS_SEGMENT_TARGET_MAX_CODES",
                DEFAULT_SEGMENT_TARGET_MAX_CODES,
            ),
            segment_break_on_commas: env_bool(
                "RUST_TTS_SEGMENT_BREAK_ON_COMMAS",
                DEFAULT_SEGMENT_BREAK_ON_COMMAS,
            ),
            incremental_enabled: env_bool("RUST_TTS_INCREMENTAL", true),
        };

        if cfg.segment_max_bytes == 0 {
            return Err("RUST_TTS_SEGMENT_MAX_BYTES must be > 0".to_string());
        }
        if !(0.0..=5.0).contains(&cfg.segment_gap_seconds) {
            return Err("RUST_TTS_SEGMENT_GAP_SECONDS must be in [0.0, 5.0]".to_string());
        }
        if cfg.max_concurrent_speech_requests == 0 {
            cfg.max_concurrent_speech_requests = 1;
        }
        if cfg.max_request_body_bytes == 0 {
            return Err("RUST_TTS_MAX_REQUEST_BODY_BYTES must be > 0".to_string());
        }
        if cfg.max_audio_sample_bytes == 0 {
            return Err("RUST_TTS_MAX_AUDIO_SAMPLE_BYTES must be > 0".to_string());
        }
        if cfg.legacy_fixed_max_codes <= 0 {
            return Err("RUST_TTS_LEGACY_MAX_CODES must be > 0".to_string());
        }
        if cfg.min_generation_codes <= 0 {
            return Err("RUST_TTS_MIN_GENERATION_CODES must be > 0".to_string());
        }
        if cfg.max_generation_codes <= 0 {
            return Err("RUST_TTS_MAX_GENERATION_CODES must be > 0".to_string());
        }
        if cfg.max_generation_codes < cfg.min_generation_codes {
            return Err(
                "RUST_TTS_MAX_GENERATION_CODES must be >= RUST_TTS_MIN_GENERATION_CODES"
                    .to_string(),
            );
        }
        if cfg.base_generation_codes < 0 {
            return Err("RUST_TTS_BASE_GENERATION_CODES must be >= 0".to_string());
        }
        if cfg.codes_per_char <= 0.0 {
            return Err("RUST_TTS_CODES_PER_CHAR must be > 0".to_string());
        }
        if cfg.vocoder_chunk_codes < 0 {
            return Err("RUST_TTS_VOCODER_CHUNK must be >= 0".to_string());
        }
        if cfg.segment_target_max_codes <= 0 {
            return Err("RUST_TTS_SEGMENT_TARGET_MAX_CODES must be > 0".to_string());
        }

        Ok(cfg)
    }

    /// Derive a per-segment character budget used by text splitting.
    /// Keeps segments inside a stable generation regime even for long articles.
    pub fn segment_max_chars(&self) -> usize {
        let target_codes = self
            .segment_target_max_codes
            .clamp(self.min_generation_codes, self.max_generation_codes);
        let available = target_codes.saturating_sub(self.base_generation_codes).max(1) as f32;
        let estimated = (available / self.codes_per_char).floor() as usize;
        estimated.max(32)
    }

    pub fn estimate_segment_max_codes(&self, input: &str) -> i64 {
        if !self.incremental_enabled {
            return self.legacy_fixed_max_codes;
        }

        let chars = input.chars().count() as f32;
        // Generation budget should be derived from text length and explicit min/max bounds.
        // `vocoder_chunk_codes` controls decode chunking only and must not truncate generation.
        ((chars * self.codes_per_char).ceil() as i64 + self.base_generation_codes)
            .max(self.min_generation_codes)
            .min(self.max_generation_codes)
    }
}

/// Server configuration populated from environment variables.
/// Matches the Python server's env var interface.
pub struct ServerConfig {
    /// Path to CustomVoice model directory (TTS_CUSTOMVOICE_MODEL_PATH)
    pub tts_customvoice_model_path: Option<String>,
    /// Path to Base model directory for voice cloning (TTS_BASE_MODEL_PATH)
    pub tts_base_model_path: Option<String>,
    /// Path to ASR model directory (ASR_MODEL_PATH)
    pub asr_model_path: Option<String>,
    /// TTS compute device preference (TTS_DEVICE): cuda, cpu, or auto
    pub tts_device: String,
    /// ASR compute device preference (ASR_DEVICE): cuda, cpu, or auto
    pub asr_device: String,
    /// Default reference audio path used when request omits audio_sample (DEFAULT_AUDIO_SAMPLE_PATH)
    pub default_audio_sample_path: Option<String>,
    /// Default transcript for the reference audio (DEFAULT_AUDIO_SAMPLE_TEXT)
    pub default_audio_sample_text: Option<String>,
    /// Default speaking instructions used when request omits instructions (DEFAULT_INSTRUCTIONS)
    pub default_instructions: Option<String>,
    /// Bind address (HOST, default: 0.0.0.0)
    pub host: String,
    /// Port (PORT, default: 38317)
    pub port: u16,
    /// Runtime controls for speech generation.
    pub speech_runtime: SpeechRuntimeConfig,
}

impl ServerConfig {
    /// Parse configuration from environment variables.
    pub fn from_env() -> Result<Self, String> {
        let config = Self {
            tts_customvoice_model_path: std::env::var("TTS_CUSTOMVOICE_MODEL_PATH")
                .ok()
                .filter(|s| !s.is_empty()),
            tts_base_model_path: std::env::var("TTS_BASE_MODEL_PATH")
                .ok()
                .filter(|s| !s.is_empty()),
            asr_model_path: std::env::var("ASR_MODEL_PATH")
                .ok()
                .filter(|s| !s.is_empty()),
            tts_device: std::env::var("TTS_DEVICE")
                .unwrap_or_else(|_| "cuda".to_string())
                .trim()
                .to_ascii_lowercase(),
            asr_device: std::env::var("ASR_DEVICE")
                .unwrap_or_else(|_| "cuda".to_string())
                .trim()
                .to_ascii_lowercase(),
            default_audio_sample_path: std::env::var("DEFAULT_AUDIO_SAMPLE_PATH")
                .ok()
                .filter(|s| !s.is_empty()),
            default_audio_sample_text: std::env::var("DEFAULT_AUDIO_SAMPLE_TEXT")
                .ok()
                .filter(|s| !s.is_empty()),
            default_instructions: std::env::var("DEFAULT_INSTRUCTIONS")
                .ok()
                .filter(|s| !s.is_empty()),
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(38317),
            speech_runtime: SpeechRuntimeConfig::from_env()?,
        };

        if config.tts_customvoice_model_path.is_none()
            && config.tts_base_model_path.is_none()
            && config.asr_model_path.is_none()
        {
            return Err(
                "At least one of TTS_CUSTOMVOICE_MODEL_PATH, TTS_BASE_MODEL_PATH, \
                 or ASR_MODEL_PATH must be set."
                    .to_string(),
            );
        }

        if config.default_audio_sample_path.is_some() && config.tts_base_model_path.is_none() {
            return Err(
                "DEFAULT_AUDIO_SAMPLE_PATH requires TTS_BASE_MODEL_PATH because default \
                 voice cloning uses the Base model."
                    .to_string(),
            );
        }

        Ok(config)
    }
}

/// OpenAI voice name -> Qwen3-TTS speaker name mapping.
fn voice_map() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("alloy", "Vivian"),
        ("ash", "Serena"),
        ("ballad", "Uncle_Fu"),
        ("coral", "Dylan"),
        ("echo", "Eric"),
        ("fable", "Ryan"),
        ("onyx", "Aiden"),
        ("nova", "Ono_Anna"),
        ("sage", "Sohee"),
        ("shimmer", "Vivian"),
        ("verse", "Ryan"),
        ("marin", "Serena"),
        ("cedar", "Aiden"),
    ])
}

/// Valid Qwen3-TTS speaker names.
fn qwen_speakers() -> HashSet<&'static str> {
    HashSet::from([
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee",
    ])
}

/// Resolve an OpenAI voice name or Qwen speaker name to a Qwen speaker.
pub fn resolve_voice(voice: &str) -> Result<String, String> {
    let speakers = qwen_speakers();
    if speakers.contains(voice) {
        return Ok(voice.to_string());
    }
    let map = voice_map();
    if let Some(&speaker) = map.get(voice.to_lowercase().as_str()) {
        return Ok(speaker.to_string());
    }
    let mut supported_openai: Vec<_> = map.keys().copied().collect();
    supported_openai.sort();
    let mut supported_qwen: Vec<_> = speakers.iter().copied().collect();
    supported_qwen.sort();
    Err(format!(
        "Unknown voice '{}'. Supported OpenAI voices: {:?}. Supported Qwen speakers: {:?}.",
        voice, supported_openai, supported_qwen
    ))
}

/// Supported audio response formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl ResponseFormat {
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/opus",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp3 => ".mp3",
            Self::Opus => ".ogg",
            Self::Aac => ".aac",
            Self::Flac => ".flac",
            Self::Wav => ".wav",
            Self::Pcm => ".pcm",
        }
    }
}
