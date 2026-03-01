use std::collections::{HashMap, HashSet};

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
    /// Port (PORT, default: 8000)
    pub port: u16,
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
                .unwrap_or(8000),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormat {
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

impl Default for ResponseFormat {
    fn default() -> Self {
        Self::Mp3
    }
}
