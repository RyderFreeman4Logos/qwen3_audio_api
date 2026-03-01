use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use qwen3_asr::inference::AsrInference;
use qwen3_tts::audio_encoder::AudioEncoder;
use qwen3_tts::inference::TTSInference;
use qwen3_tts::speaker_encoder::SpeakerEncoder;
use tokio::sync::{Semaphore, SemaphorePermit};

use crate::config::SpeechRuntimeConfig;

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
    current_task_total_segments: AtomicU64,
    current_task_completed_segments: AtomicU64,
}

impl SpeechTaskControl {
    pub fn new() -> Self {
        Self {
            next_task_id: AtomicU64::new(1),
            current_task_id: AtomicU64::new(0),
            cancelled_up_to_task_id: AtomicU64::new(0),
            current_task_total_segments: AtomicU64::new(0),
            current_task_completed_segments: AtomicU64::new(0),
        }
    }

    pub fn alloc_task_id(&self) -> u64 {
        self.next_task_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn set_current_task_id(&self, task_id: u64) {
        self.current_task_id.store(task_id, Ordering::SeqCst);
        self.current_task_total_segments.store(0, Ordering::SeqCst);
        self.current_task_completed_segments
            .store(0, Ordering::SeqCst);
    }

    pub fn clear_current_task_id(&self, task_id: u64) {
        let current = self.current_task_id.load(Ordering::SeqCst);
        if current == task_id {
            self.current_task_id.store(0, Ordering::SeqCst);
            self.current_task_total_segments.store(0, Ordering::SeqCst);
            self.current_task_completed_segments
                .store(0, Ordering::SeqCst);
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

    pub fn set_current_task_total_segments(&self, task_id: u64, total_segments: u64) {
        if self.current_task_id.load(Ordering::SeqCst) != task_id {
            return;
        }
        self.current_task_total_segments
            .store(total_segments, Ordering::SeqCst);
        let completed = self.current_task_completed_segments.load(Ordering::SeqCst);
        if completed > total_segments {
            self.current_task_completed_segments
                .store(total_segments, Ordering::SeqCst);
        }
    }

    pub fn set_current_task_completed_segments(&self, task_id: u64, completed_segments: u64) {
        if self.current_task_id.load(Ordering::SeqCst) != task_id {
            return;
        }
        let total = self.current_task_total_segments.load(Ordering::SeqCst);
        let bounded = if total > 0 {
            completed_segments.min(total)
        } else {
            completed_segments
        };
        self.current_task_completed_segments
            .store(bounded, Ordering::SeqCst);
    }

    pub fn current_task_progress(&self) -> Option<(u64, u64)> {
        self.current_task_id().map(|_| {
            (
                self.current_task_completed_segments.load(Ordering::SeqCst),
                self.current_task_total_segments.load(Ordering::SeqCst),
            )
        })
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
    pub speech_runtime: SpeechRuntimeConfig,
    speech_inference_slots: Semaphore,
    queued_speech_waiters: AtomicUsize,
}

impl AppContext {
    /// Acquire the models lock and recover automatically from poison.
    ///
    /// A previous panic while holding the lock should not permanently block
    /// all subsequent requests; we log and continue with the contained state.
    pub fn lock_models_recover(&self) -> MutexGuard<'_, Models> {
        match self.models.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::warn!("recovering from poisoned models mutex");
                self.models.clear_poison();
                poisoned.into_inner()
            }
        }
    }

    /// Acquire an inference slot with bounded queueing.
    pub async fn acquire_speech_slot(&self) -> Result<SemaphorePermit<'_>, AcquireSpeechSlotError> {
        let waiting_now = self.queued_speech_waiters.fetch_add(1, Ordering::SeqCst) + 1;
        if waiting_now > self.speech_runtime.max_queued_speech_requests {
            self.queued_speech_waiters.fetch_sub(1, Ordering::SeqCst);
            return Err(AcquireSpeechSlotError::QueueFull {
                queued: waiting_now,
                limit: self.speech_runtime.max_queued_speech_requests,
            });
        }

        let permit = self
            .speech_inference_slots
            .acquire()
            .await
            .map_err(|_| AcquireSpeechSlotError::GateClosed)?;
        self.queued_speech_waiters.fetch_sub(1, Ordering::SeqCst);
        Ok(permit)
    }
}

#[derive(Debug)]
pub enum AcquireSpeechSlotError {
    QueueFull { queued: usize, limit: usize },
    GateClosed,
}

/// Shared application state threaded through all handlers.
/// The Mutex provides the same single-request-at-a-time guarantee
/// as the Python server's threading.Lock.
pub type AppState = Arc<AppContext>;

pub fn new_app_state(models: Models, speech_runtime: SpeechRuntimeConfig) -> AppState {
    let max_concurrency = speech_runtime.max_concurrent_speech_requests;
    Arc::new(AppContext {
        models: Mutex::new(models),
        speech_tasks: SpeechTaskControl::new(),
        speech_runtime,
        speech_inference_slots: Semaphore::new(max_concurrency),
        queued_speech_waiters: AtomicUsize::new(0),
    })
}

#[cfg(test)]
mod tests {
    use super::{new_app_state, Models, SpeechTaskControl};
    use crate::config::SpeechRuntimeConfig;

    fn empty_models() -> Models {
        Models {
            custom_voice: None,
            base_model: None,
            speaker_encoder: None,
            audio_encoder: None,
            asr: None,
            default_audio_sample_wav_bytes: None,
            default_audio_sample_text: None,
            default_instructions: None,
        }
    }

    #[test]
    fn lock_models_recover_works_after_poison() {
        let runtime = SpeechRuntimeConfig::from_env().expect("default runtime config should parse");
        let state = new_app_state(empty_models(), runtime);

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let state = state.clone();
            move || {
                let _guard = state.models.lock().expect("lock should succeed");
                panic!("force poison");
            }
        }));

        let guard = state.lock_models_recover();
        assert!(guard.custom_voice.is_none());
    }

    #[test]
    fn cancel_current_cancels_up_to_active_task_boundary() {
        let control = SpeechTaskControl::new();
        let first = control.alloc_task_id();
        let second = control.alloc_task_id();
        control.set_current_task_id(second);

        assert_eq!(control.cancel_current(), Some(second));
        assert!(control.is_cancelled(first));
        assert!(control.is_cancelled(second));
        assert!(!control.is_cancelled(second + 1));
    }

    #[test]
    fn cancel_all_does_not_kill_future_tasks() {
        let control = SpeechTaskControl::new();
        let _ = control.alloc_task_id();
        let second = control.alloc_task_id();

        let cancelled_up_to = control.cancel_all();
        assert_eq!(cancelled_up_to, second);
        assert!(control.is_cancelled(second));

        let future = control.alloc_task_id();
        assert!(!control.is_cancelled(future));
    }

    #[test]
    fn current_task_progress_tracks_active_task_only() {
        let control = SpeechTaskControl::new();
        let first = control.alloc_task_id();
        let second = control.alloc_task_id();

        control.set_current_task_id(second);
        control.set_current_task_total_segments(second, 10);
        control.set_current_task_completed_segments(second, 3);
        assert_eq!(control.current_task_progress(), Some((3, 10)));

        control.set_current_task_completed_segments(first, 8);
        assert_eq!(control.current_task_progress(), Some((3, 10)));

        control.clear_current_task_id(second);
        assert_eq!(control.current_task_progress(), None);
    }
}
