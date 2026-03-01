use std::sync::atomic::{AtomicU64, Ordering};
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

/// Global speech task control for cancellation.
pub struct SpeechTaskControl {
    next_task_id: AtomicU64,
    current_task_id: AtomicU64,
    cancelled_up_to_task_id: AtomicU64,
}

impl SpeechTaskControl {
    pub fn new() -> Self {
        Self {
            next_task_id: AtomicU64::new(1),
            current_task_id: AtomicU64::new(0),
            cancelled_up_to_task_id: AtomicU64::new(0),
        }
    }

    pub fn alloc_task_id(&self) -> u64 {
        self.next_task_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn set_current_task_id(&self, task_id: u64) {
        self.current_task_id.store(task_id, Ordering::SeqCst);
    }

    pub fn clear_current_task_id(&self, task_id: u64) {
        let current = self.current_task_id.load(Ordering::SeqCst);
        if current == task_id {
            self.current_task_id.store(0, Ordering::SeqCst);
        }
    }

    pub fn current_task_id(&self) -> Option<u64> {
        let id = self.current_task_id.load(Ordering::SeqCst);
        if id == 0 {
            None
        } else {
            Some(id)
        }
    }

    pub fn cancel_current(&self) -> Option<u64> {
        let current = self.current_task_id();
        if let Some(task_id) = current {
            self.cancel_up_to(task_id);
        }
        current
    }

    pub fn cancel_all(&self) -> u64 {
        let max_existing_task_id = self.next_task_id.load(Ordering::SeqCst).saturating_sub(1);
        self.cancel_up_to(max_existing_task_id);
        max_existing_task_id
    }

    pub fn is_cancelled(&self, task_id: u64) -> bool {
        task_id <= self.cancelled_up_to_task_id.load(Ordering::SeqCst)
    }

    fn cancel_up_to(&self, task_id: u64) {
        let mut prev = self.cancelled_up_to_task_id.load(Ordering::SeqCst);
        while task_id > prev {
            match self.cancelled_up_to_task_id.compare_exchange(
                prev,
                task_id,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }
    }
}

pub struct AppContext {
    pub models: Mutex<Models>,
    pub speech_tasks: SpeechTaskControl,
}

/// Shared application state threaded through all handlers.
/// The Mutex provides the same single-request-at-a-time guarantee
/// as the Python server's threading.Lock.
pub type AppState = Arc<AppContext>;

pub fn new_app_state(models: Models) -> AppState {
    Arc::new(AppContext {
        models: Mutex::new(models),
        speech_tasks: SpeechTaskControl::new(),
    })
}
