"""FastAPI server implementing OpenAI-compatible audio APIs with Qwen3-TTS/ASR.

Supports long-text synthesis via automatic segmentation, segment-level
progress tracking, and task cancellation — matching the Rust API surface
so the existing Rust client works without changes.
"""

import io
import logging
import math
import os
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice mapping
# ---------------------------------------------------------------------------
VOICE_MAP: dict[str, str] = {
    "alloy": "Vivian",
    "ash": "Serena",
    "ballad": "Uncle_Fu",
    "coral": "Dylan",
    "echo": "Eric",
    "fable": "Ryan",
    "onyx": "Aiden",
    "nova": "Ono_Anna",
    "sage": "Sohee",
    "shimmer": "Vivian",
    "verse": "Ryan",
    "marin": "Serena",
    "cedar": "Aiden",
}

QWEN_SPEAKERS: set[str] = {
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
}


class ResponseFormat(str, Enum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"
    pcm = "pcm"


class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "alloy"
    response_format: ResponseFormat = ResponseFormat.mp3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str = "Auto"
    instructions: str | None = None
    audio_sample: str | None = None
    audio_sample_text: str | None = None


# ---------------------------------------------------------------------------
# Runtime configuration (matches Rust env vars)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "")
    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name, "")
    try:
        return float(val.strip())
    except (ValueError, AttributeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


class SpeechRuntimeConfig:
    """Runtime controls for speech synthesis, loaded from environment."""

    def __init__(self) -> None:
        self.segment_max_bytes = _env_int("RUST_TTS_SEGMENT_MAX_BYTES", 1024)
        self.segment_gap_seconds = _env_float("RUST_TTS_SEGMENT_GAP_SECONDS", 0.12)
        self.segment_break_on_commas = _env_bool(
            "RUST_TTS_SEGMENT_BREAK_ON_COMMAS", False
        )
        self.segment_target_max_codes = _env_int(
            "RUST_TTS_SEGMENT_TARGET_MAX_CODES", 640
        )
        self.base_generation_codes = _env_int("RUST_TTS_BASE_GENERATION_CODES", 96)
        self.min_generation_codes = _env_int("RUST_TTS_MIN_GENERATION_CODES", 192)
        self.max_generation_codes = _env_int("RUST_TTS_MAX_GENERATION_CODES", 2048)
        self.codes_per_char = _env_float("RUST_TTS_CODES_PER_CHAR", 3.8)
        self.generation_temperature = _env_float("RUST_TTS_TEMPERATURE", 0.9)
        self.generation_top_k = _env_int("RUST_TTS_TOP_K", 50)
        self.max_concurrent_speech_requests = _env_int(
            "RUST_TTS_MAX_CONCURRENT_REQUESTS", 1
        )
        self.max_queued_speech_requests = _env_int("RUST_TTS_MAX_QUEUED_REQUESTS", 8)

    def segment_max_chars(self) -> int:
        target = max(
            self.min_generation_codes,
            min(self.segment_target_max_codes, self.max_generation_codes),
        )
        available = max(target - self.base_generation_codes, 1)
        estimated = int(available / self.codes_per_char)
        return max(estimated, 32)


# ---------------------------------------------------------------------------
# Text segmentation (ported from Rust speech.rs)
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARIES = frozenset("。！？；.!?;\n\r")
_COMMA_BOUNDARIES = frozenset("，,、")


def _is_preferred_boundary(ch: str, split_on_commas: bool) -> bool:
    return ch in _SENTENCE_BOUNDARIES or (
        split_on_commas and ch in _COMMA_BOUNDARIES
    )


def split_text_for_tts(
    text: str,
    max_bytes: int,
    max_chars: int,
    split_on_commas: bool,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    normalized = text.replace("\r\n", "\n")
    chunks: list[str] = []
    for paragraph in normalized.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        _split_block(paragraph, max_bytes, max_chars, split_on_commas, chunks)

    return chunks if chunks else [normalized]


def _split_block(
    block: str,
    max_bytes: int,
    max_chars: int,
    split_on_commas: bool,
    out: list[str],
) -> None:
    start = 0
    while start < len(block):
        # Skip whitespace
        while start < len(block) and block[start].isspace():
            start += 1
        if start >= len(block):
            return

        # Find budget limit
        end = start
        byte_count = 0
        char_count = 0
        for i, ch in enumerate(block[start:]):
            ch_bytes = len(ch.encode("utf-8"))
            if byte_count + ch_bytes > max_bytes or char_count + 1 > max_chars:
                break
            byte_count += ch_bytes
            char_count += 1
            end = start + i + 1

        hard_limit = end if end > start else start + 1

        if hard_limit >= len(block):
            split_pos = len(block)
        else:
            # Find last preferred boundary within budget
            last_boundary = None
            for i, ch in enumerate(block[start:hard_limit]):
                if _is_preferred_boundary(ch, split_on_commas):
                    last_boundary = start + i + 1
            split_pos = last_boundary if last_boundary is not None else hard_limit

        if split_pos <= start:
            split_pos = start + 1

        chunk = block[start:split_pos].strip()
        if chunk:
            out.append(chunk)
        start = split_pos


# ---------------------------------------------------------------------------
# Task control (cancellation + progress)
# ---------------------------------------------------------------------------

class SpeechTaskControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self._current_id = 0
        self._cancelled_up_to = 0
        self._total_segments = 0
        self._completed_segments = 0

    def alloc_task_id(self) -> int:
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            return tid

    def set_current(self, task_id: int, total_segments: int) -> None:
        with self._lock:
            self._current_id = task_id
            self._total_segments = total_segments
            self._completed_segments = 0

    def clear_current(self, task_id: int) -> None:
        with self._lock:
            if self._current_id == task_id:
                self._current_id = 0
                self._total_segments = 0
                self._completed_segments = 0

    def set_completed_segments(self, task_id: int, completed: int) -> None:
        with self._lock:
            if self._current_id == task_id:
                self._completed_segments = min(completed, self._total_segments)

    def is_cancelled(self, task_id: int) -> bool:
        with self._lock:
            return task_id <= self._cancelled_up_to

    def cancel_current(self) -> int | None:
        with self._lock:
            if self._current_id == 0:
                return None
            tid = self._current_id
            self._cancelled_up_to = max(self._cancelled_up_to, tid)
            return tid

    def cancel_all(self) -> int:
        with self._lock:
            max_id = self._next_id - 1
            self._cancelled_up_to = max(self._cancelled_up_to, max_id)
            return max_id

    def status(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "current_task_id": self._current_id if self._current_id else None,
                "completed_segments": self._completed_segments,
                "total_segments": self._total_segments,
            }


# ---------------------------------------------------------------------------
# Audio encoding helpers
# ---------------------------------------------------------------------------

CONTENT_TYPES: dict[ResponseFormat, str] = {
    ResponseFormat.mp3: "audio/mpeg",
    ResponseFormat.opus: "audio/opus",
    ResponseFormat.aac: "audio/aac",
    ResponseFormat.flac: "audio/flac",
    ResponseFormat.wav: "audio/wav",
    ResponseFormat.pcm: "audio/pcm",
}


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def _encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC")
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    return int16.tobytes()


def _encode_with_ffmpeg(
    audio: np.ndarray,
    sample_rate: int,
    fmt: ResponseFormat,
) -> bytes:
    wav_bytes = _encode_wav(audio, sample_rate)
    codec_map = {
        ResponseFormat.mp3: ("libmp3lame", "mp3"),
        ResponseFormat.opus: ("libopus", "ogg"),
        ResponseFormat.aac: ("aac", "adts"),
    }
    codec, container = codec_map[fmt]
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-acodec",
            codec,
            "-f",
            container,
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise HTTPException(
            status_code=500, detail=f"ffmpeg encoding failed: {stderr}"
        )
    return result.stdout


def encode_audio(
    audio: np.ndarray, sample_rate: int, fmt: ResponseFormat
) -> bytes:
    if fmt == ResponseFormat.wav:
        return _encode_wav(audio, sample_rate)
    if fmt == ResponseFormat.flac:
        return _encode_flac(audio, sample_rate)
    if fmt == ResponseFormat.pcm:
        return _encode_pcm(audio)
    return _encode_with_ffmpeg(audio, sample_rate, fmt)


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio
    from scipy.signal import resample

    new_length = int(len(audio) / speed)
    if new_length == 0:
        return audio
    return resample(audio, new_length).astype(np.float32)


def resolve_voice(voice: str) -> str:
    if voice in QWEN_SPEAKERS:
        return voice
    mapped = VOICE_MAP.get(voice.lower())
    if mapped is not None:
        return mapped
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown voice '{voice}'. "
            f"Supported OpenAI voices: {sorted(VOICE_MAP.keys())}. "
            f"Supported Qwen speakers: {sorted(QWEN_SPEAKERS)}."
        ),
    )


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_inference_lock = threading.Lock()
_inference_semaphore: threading.Semaphore | None = None
_task_control = SpeechTaskControl()
_runtime_config: SpeechRuntimeConfig | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _inference_semaphore, _runtime_config

    _runtime_config = SpeechRuntimeConfig()
    _inference_semaphore = threading.Semaphore(
        _runtime_config.max_concurrent_speech_requests
    )

    model_path = os.environ.get("TTS_CUSTOMVOICE_MODEL_PATH", "")
    base_model_path = os.environ.get("TTS_BASE_MODEL_PATH", "")
    asr_model_path = os.environ.get("ASR_MODEL_PATH", "")
    device = os.environ.get("QWEN_TTS_DEVICE", "cuda:0")
    dtype_name = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    attn_impl = os.environ.get("QWEN_TTS_ATTN", "flash_attention_2")

    if not model_path and not base_model_path and not asr_model_path:
        raise RuntimeError(
            "At least one of TTS_CUSTOMVOICE_MODEL_PATH, TTS_BASE_MODEL_PATH, "
            "or ASR_MODEL_PATH must be set."
        )

    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash-attn is not installed, disabling flash_attention_2. "
                "Install with: pip install -U flash-attn --no-build-isolation"
            )
            attn_impl = ""

    kwargs: dict[str, object] = {"device_map": device, "dtype": dtype}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    app.state.model = None
    if model_path:
        model_path = os.path.abspath(model_path)
        old_cwd = os.getcwd()
        os.chdir(model_path)
        logger.info(
            "Loading custom-voice model from %s on %s (%s, attn=%s)",
            model_path, device, dtype_name, attn_impl,
        )
        try:
            app.state.model = Qwen3TTSModel.from_pretrained(".", **kwargs)
            logger.info("Custom-voice model loaded successfully")
        finally:
            os.chdir(old_cwd)

    app.state.base_model = None
    if base_model_path:
        base_model_path = os.path.abspath(base_model_path)
        old_cwd = os.getcwd()
        os.chdir(base_model_path)
        logger.info(
            "Loading base model from %s on %s (%s, attn=%s)",
            base_model_path, device, dtype_name, attn_impl,
        )
        try:
            app.state.base_model = Qwen3TTSModel.from_pretrained(".", **kwargs)
            logger.info("Base model loaded successfully")
        finally:
            os.chdir(old_cwd)

    app.state.asr_model = None
    if asr_model_path:
        logger.info(
            "Loading ASR model %s on %s (%s, attn=%s)",
            asr_model_path, device, dtype_name, attn_impl,
        )
        asr_kwargs: dict[str, object] = {
            "device_map": device,
            "dtype": dtype,
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        if attn_impl:
            asr_kwargs["attn_implementation"] = attn_impl
        app.state.asr_model = Qwen3ASRModel.from_pretrained(
            asr_model_path, **asr_kwargs
        )
        logger.info("ASR model loaded successfully")

    logger.info(
        "Runtime config: segment_max_bytes=%d, segment_max_chars=%d, "
        "target_max_codes=%d, split_on_commas=%s, temperature=%.2f, top_k=%d",
        _runtime_config.segment_max_bytes,
        _runtime_config.segment_max_chars(),
        _runtime_config.segment_target_max_codes,
        _runtime_config.segment_break_on_commas,
        _runtime_config.generation_temperature,
        _runtime_config.generation_top_k,
    )

    yield


app = FastAPI(
    title="Qwen3-TTS OpenAI-Compatible API",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Speech generation with segmentation
# ---------------------------------------------------------------------------

def _generate_segment(
    model: Qwen3TTSModel,
    text: str,
    *,
    is_voice_clone: bool,
    language: str,
    ref_audio: object = None,
    ref_text: str | None = None,
    use_icl: bool = False,
    speaker: str = "",
    instruct: str = "",
) -> tuple[np.ndarray, int]:
    """Generate speech for a single text segment."""
    if is_voice_clone:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=not use_icl,
        )
    else:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
    return wavs[0], sr


def _synthesize_with_segments(
    model: Qwen3TTSModel,
    segments: list[str],
    task_id: int,
    *,
    is_voice_clone: bool,
    language: str,
    ref_audio: object = None,
    ref_text: str | None = None,
    use_icl: bool = False,
    speaker: str = "",
    instruct: str = "",
    gap_seconds: float = 0.12,
) -> tuple[np.ndarray, int]:
    """Generate speech segment-by-segment, merging with silence gaps."""
    assert _runtime_config is not None
    merged: list[np.ndarray] = []
    sample_rate = 24000

    for idx, segment in enumerate(segments):
        if _task_control.is_cancelled(task_id):
            raise HTTPException(
                status_code=400,
                detail=f"speech task {task_id} was cancelled",
            )

        logger.info(
            "speech task %d segment %d/%d: chars=%d",
            task_id, idx + 1, len(segments), len(segment),
        )

        audio, sr = _generate_segment(
            model,
            segment,
            is_voice_clone=is_voice_clone,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            use_icl=use_icl,
            speaker=speaker,
            instruct=instruct,
        )
        sample_rate = sr

        if merged:
            gap_samples = int(sr * gap_seconds)
            merged.append(np.zeros(gap_samples, dtype=np.float32))
        merged.append(audio.astype(np.float32))

        _task_control.set_completed_segments(task_id, idx + 1)

    if not merged:
        return np.zeros(sample_rate * 2, dtype=np.float32), sample_rate

    return np.concatenate(merged), sample_rate


@app.post("/v1/audio/speech")
async def create_speech(raw_request: Request) -> Response:
    """Generate audio from text (OpenAI-compatible).

    Accepts JSON or multipart/form-data. Automatically segments long text.
    """
    assert _runtime_config is not None

    req_content_type = raw_request.headers.get("content-type", "")

    if "multipart/form-data" in req_content_type:
        form = await raw_request.form()
        input_text = str(form.get("input", ""))
        if not input_text:
            raise HTTPException(status_code=422, detail="'input' field is required")
        voice = str(form.get("voice", "alloy"))
        try:
            fmt = ResponseFormat(str(form.get("response_format", "mp3")))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        speed = float(form.get("speed", "1.0"))
        if not 0.25 <= speed <= 4.0:
            raise HTTPException(
                status_code=422,
                detail="'speed' must be between 0.25 and 4.0",
            )
        language = str(form.get("language", "Auto"))
        instr_val = form.get("instructions")
        instructions = str(instr_val) if instr_val is not None else None
        sample_text_val = form.get("audio_sample_text")
        audio_sample_text = (
            str(sample_text_val) if sample_text_val is not None else None
        )
        audio_upload = form.get("audio_sample")
        ref_audio: tuple[np.ndarray, int] | str | None = None
        if audio_upload is not None and hasattr(audio_upload, "read"):
            audio_bytes = await audio_upload.read()
            audio_arr, audio_sr = sf.read(io.BytesIO(audio_bytes))
            ref_audio = (audio_arr.astype(np.float32), int(audio_sr))
        elif audio_upload is not None:
            ref_audio = str(audio_upload)
    else:
        request = SpeechRequest(**(await raw_request.json()))
        input_text = request.input
        voice = request.voice
        fmt = request.response_format
        speed = request.speed
        language = request.language
        instructions = request.instructions
        audio_sample_text = request.audio_sample_text
        ref_audio = request.audio_sample

    if not input_text.strip():
        raise HTTPException(status_code=422, detail="'input' field is required")

    # Load default reference audio if none provided
    default_ref_path = os.environ.get("DEFAULT_AUDIO_SAMPLE_PATH", "")
    default_ref_text = os.environ.get("DEFAULT_AUDIO_SAMPLE_TEXT", "")
    if ref_audio is None and default_ref_path and os.path.isfile(default_ref_path):
        ref_audio = default_ref_path
        if not audio_sample_text and default_ref_text:
            audio_sample_text = default_ref_text

    # Segment input text
    segments = split_text_for_tts(
        input_text,
        _runtime_config.segment_max_bytes,
        _runtime_config.segment_max_chars(),
        _runtime_config.segment_break_on_commas,
    )

    task_id = _task_control.alloc_task_id()
    _task_control.set_current(task_id, len(segments))

    logger.info(
        "speech task %d accepted: input_chars=%d, segments=%d",
        task_id, len(input_text), len(segments),
    )

    try:
        if ref_audio is not None:
            base_model: Qwen3TTSModel | None = app.state.base_model
            if base_model is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "audio_sample requires a base model. "
                        "Set TTS_BASE_MODEL_PATH to enable voice cloning."
                    ),
                )
            use_icl = audio_sample_text is not None
            with _inference_lock:
                audio, sr = _synthesize_with_segments(
                    base_model,
                    segments,
                    task_id,
                    is_voice_clone=True,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=audio_sample_text,
                    use_icl=use_icl,
                    gap_seconds=_runtime_config.segment_gap_seconds,
                )
        else:
            model: Qwen3TTSModel | None = app.state.model
            if model is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Custom-voice model is not loaded. "
                        "Set TTS_CUSTOMVOICE_MODEL_PATH to enable speaker voices, "
                        "or provide audio_sample to use voice cloning."
                    ),
                )
            speaker = resolve_voice(voice)
            instruct = instructions or ""
            with _inference_lock:
                audio, sr = _synthesize_with_segments(
                    model,
                    segments,
                    task_id,
                    is_voice_clone=False,
                    language=language,
                    speaker=speaker,
                    instruct=instruct,
                    gap_seconds=_runtime_config.segment_gap_seconds,
                )

        audio = apply_speed(audio, speed)
        data = encode_audio(audio, sr, fmt)
        return Response(content=data, media_type=CONTENT_TYPES[fmt])
    finally:
        _task_control.clear_current(task_id)


# ---------------------------------------------------------------------------
# Cancel / status endpoints (Rust-client compatible)
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech/cancel-current")
async def cancel_current() -> JSONResponse:
    tid = _task_control.cancel_current()
    return JSONResponse({
        "ok": True,
        "message": (
            f"Cancellation requested for current speech task {tid}"
            if tid
            else "No running speech task to cancel"
        ),
        "current_task_id": _task_control.status()["current_task_id"],
        "cancelled_up_to_task_id": tid or 0,
    })


@app.post("/v1/audio/speech/cancel-all")
async def cancel_all() -> JSONResponse:
    cancelled_up_to = _task_control.cancel_all()
    return JSONResponse({
        "ok": True,
        "message": (
            f"Cancellation requested for all speech tasks up to {cancelled_up_to}"
        ),
        "current_task_id": _task_control.status()["current_task_id"],
        "cancelled_up_to_task_id": cancelled_up_to,
    })


@app.get("/v1/audio/speech/status")
async def speech_status() -> JSONResponse:
    return JSONResponse(_task_control.status())


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def convert_audio_to_wav(audio_bytes: bytes, suffix: str = ".mp3") -> str:
    if suffix.lower() == ".wav":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            wav_file.write(audio_bytes)
            return wav_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src_file:
        src_file.write(audio_bytes)
        src_path = src_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
        wav_path = wav_file.name

    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            wav_path,
        ],
        capture_output=True,
    )
    os.unlink(src_path)
    if result.returncode != 0:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        stderr = result.stderr.decode(errors="replace")
        raise HTTPException(
            status_code=500, detail=f"ffmpeg audio conversion failed: {stderr}"
        )
    return wav_path


class TranscriptionResponse(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="qwen3-asr"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> TranscriptionResponse | Response:
    asr_model: Qwen3ASRModel | None = app.state.asr_model
    if asr_model is None:
        raise HTTPException(
            status_code=400,
            detail="ASR model is not loaded. Set ASR_MODEL_PATH to enable transcription.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file")

    filename = file.filename or "audio.mp3"
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".mp3"

    wav_path: str | None = None
    try:
        wav_path = convert_audio_to_wav(audio_bytes, suffix=suffix)
        with _inference_lock:
            results = asr_model.transcribe(audio=wav_path, language=language)
        text = results[0].text if results else ""
        if response_format == "text":
            return Response(content=text, media_type="text/plain")
        return TranscriptionResponse(text=text)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    models = []
    if app.state.model is not None or app.state.base_model is not None:
        models.append({"id": "qwen3-tts", "object": "model", "owned_by": "qwen"})
    if app.state.asr_model is not None:
        models.append({"id": "qwen3-asr", "object": "model", "owned_by": "qwen"})
    return {"object": "list", "data": models}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "38317"))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=host, port=port)
