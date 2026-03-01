use std::sync::{Arc, Mutex};

use qwen3_asr::inference::AsrInference;
use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;

/// Models that may be loaded at startup.
pub struct Models {
    /// CustomVoice TTS model (for predefined speakers)
    pub custom_voice: Option<TTSInference>,
    /// Base TTS model (for voice cloning)
    pub base_model: Option<TTSInference>,
    /// Speaker encoder for x-vector extraction (loaded from base model weights)
    pub speaker_encoder: Option<SpeakerEncoder>,
    /// Audio encoder for ICL voice cloning (loaded from base model's speech_tokenizer)
    pub audio_encoder: Option<AudioEncoder>,
    /// ASR model
    pub asr: Option<AsrInference>,
    /// Optional default reference audio bytes in WAV format.
    pub default_audio_sample_wav_bytes: Option<Vec<u8>>,
    /// Optional transcript paired with the default reference audio.
    pub default_audio_sample_text: Option<String>,
    /// Optional default speaking instructions for CustomVoice requests.
    pub default_instructions: Option<String>,
}

/// Shared application state threaded through all handlers.
/// The Mutex provides the same single-request-at-a-time guarantee
/// as the Python server's threading.Lock.
pub type AppState = Arc<Mutex<Models>>;

pub fn new_app_state(models: Models) -> AppState {
    Arc::new(Mutex::new(models))
}
