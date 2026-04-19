"""FastAPI server implementing OpenAI-compatible audio APIs with Qwen3-TTS/ASR.

Supports long-text synthesis via automatic segmentation, segment-level
progress tracking, and task cancellation — matching the Rust API surface
so the existing Rust client works without changes.
"""

import asyncio
import io
import logging
import os
import struct
import subprocess
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import blake2s
from time import perf_counter
from typing import AsyncGenerator, AsyncIterator

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

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


@dataclass(slots=True)
class ParsedSpeechRequest:
    input_text: str
    voice: str
    response_format: ResponseFormat
    speed: float
    language: str
    instructions: str | None
    audio_sample_text: str | None
    ref_audio: tuple[np.ndarray, int] | str | None


@dataclass(slots=True)
class SpeechSynthesisPlan:
    task_id: int
    segments: list[str]
    speed: float
    response_format: ResponseFormat
    model: Qwen3TTSModel
    is_voice_clone: bool
    language: str
    ref_audio: object = None
    ref_text: str | None = None
    use_icl: bool = False
    voice_clone_prompt: list[VoiceClonePromptItem] | None = None
    speaker: str = ""
    instruct: str = ""


@dataclass(slots=True)
class VoiceArtifacts:
    normalized_ref_audio: tuple[np.ndarray, int]
    ref_code: torch.Tensor
    ref_spk_embedding: torch.Tensor


@dataclass(slots=True)
class SegmentTalkerResult:
    codes_for_decode: torch.Tensor
    ref_code_length: int = 0


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
    return ch in _SENTENCE_BOUNDARIES or (split_on_commas and ch in _COMMA_BOUNDARIES)


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

    def cancel_task(self, task_id: int) -> bool:
        with self._lock:
            if self._current_id != task_id:
                return False
            self._cancelled_up_to = max(self._cancelled_up_to, task_id)
            return True

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

TTS_SAMPLE_RATE = 24000
WAV_BITS_PER_SAMPLE = 16
WAV_STREAMING_DATA_SIZE = 0xFFFFFFFF


def _audio_channels(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else int(audio.shape[1])


def _wav_header(sample_rate: int, channels: int, data_size: int) -> bytes:
    block_align = channels * (WAV_BITS_PER_SAMPLE // 8)
    byte_rate = sample_rate * block_align
    riff_size = (
        WAV_STREAMING_DATA_SIZE
        if data_size == WAV_STREAMING_DATA_SIZE
        else 36 + data_size
    )
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size & 0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        WAV_BITS_PER_SAMPLE,
        b"data",
        data_size & 0xFFFFFFFF,
    )


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = _encode_pcm(audio)
    return _wav_header(sample_rate, _audio_channels(audio), len(pcm)) + pcm


def _encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC")
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    return int16.tobytes()


def _decode_audio_sample(
    audio_bytes: bytes, *, target_sr: int
) -> tuple[np.ndarray, int]:
    try:
        audio_arr, audio_sr = sf.read(io.BytesIO(audio_bytes))
    except sf.LibsndfileError as exc:
        logger.info("audio decode fallback via ffmpeg (reason: %s)", exc)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-ar",
                    str(target_sr),
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError as ffmpeg_exc:
            raise HTTPException(status_code=400, detail=str(ffmpeg_exc)) from ffmpeg_exc
        except subprocess.TimeoutExpired as ffmpeg_exc:
            raise HTTPException(
                status_code=400, detail="ffmpeg timeout"
            ) from ffmpeg_exc

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise HTTPException(
                status_code=400,
                detail=stderr or "ffmpeg audio decode failed",
            )

        try:
            audio_arr, audio_sr = sf.read(io.BytesIO(result.stdout))
        except sf.LibsndfileError as transcode_exc:
            raise HTTPException(
                status_code=400, detail=str(transcode_exc)
            ) from transcode_exc

    return audio_arr.astype(np.float32), int(audio_sr)


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
    try:
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
            timeout=30,
        )
    except subprocess.TimeoutExpired as ffmpeg_exc:
        raise HTTPException(status_code=400, detail="ffmpeg timeout") from ffmpeg_exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise HTTPException(status_code=500, detail=f"ffmpeg encoding failed: {stderr}")
    return result.stdout


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: ResponseFormat) -> bytes:
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
_talker_lock = threading.Lock()
_vocoder_lock = threading.Lock()
_task_control = SpeechTaskControl()
_runtime_config: SpeechRuntimeConfig | None = None
_voice_artifact_cache: OrderedDict[str, VoiceArtifacts] = OrderedDict()
_voice_artifact_cache_lock = threading.Lock()
_segment_pipeline_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="qwen3-segment-pipeline",
)


def _voice_cache_capacity() -> int:
    return max(_env_int("QWEN_TTS_VOICE_CACHE_SIZE", 8), 1)


def _clear_voice_artifact_cache() -> None:
    with _voice_artifact_cache_lock:
        _voice_artifact_cache.clear()


def _voice_artifact_cache_keys_for_tests() -> list[str]:
    with _voice_artifact_cache_lock:
        return list(_voice_artifact_cache.keys())


def _serialize_voice_cache_input(ref_audio: object) -> bytes:
    if (
        isinstance(ref_audio, tuple)
        and len(ref_audio) == 2
        and isinstance(ref_audio[0], np.ndarray)
    ):
        audio = np.ascontiguousarray(ref_audio[0].astype(np.float32, copy=False))
        sample_rate = int(ref_audio[1])
        header = (
            f"ndarray:{sample_rate}:{audio.dtype.str}:{audio.ndim}:{audio.shape}"
        ).encode("utf-8")
        return header + b"\0" + audio.tobytes()

    if isinstance(ref_audio, str):
        if os.path.isfile(ref_audio):
            with open(ref_audio, "rb") as fh:
                return b"file\0" + fh.read()
        return b"string\0" + ref_audio.encode("utf-8")

    raise TypeError(f"Unsupported voice-clone reference audio: {type(ref_audio)!r}")


def _voice_artifact_cache_key(
    ref_audio: object,
    *,
    ref_text: str | None,
    use_icl: bool,
) -> str:
    hasher = blake2s()
    hasher.update(_serialize_voice_cache_input(ref_audio))
    hasher.update(b"\0")
    hasher.update(b"icl=1" if use_icl else b"icl=0")
    hasher.update(b"\0")
    hasher.update((ref_text or "").encode("utf-8"))
    return hasher.hexdigest()


def _compute_voice_artifacts(
    model: Qwen3TTSModel,
    ref_audio: object,
) -> VoiceArtifacts:
    prompt_item = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text="__voice_artifact_cache__",
        x_vector_only_mode=False,
    )[0]
    normalized_ref_audio = model._normalize_audio_inputs(ref_audio)[0]
    return VoiceArtifacts(
        normalized_ref_audio=normalized_ref_audio,
        ref_code=prompt_item.ref_code,
        ref_spk_embedding=prompt_item.ref_spk_embedding,
    )


def _build_voice_clone_prompt(
    artifacts: VoiceArtifacts,
    *,
    ref_text: str | None,
    use_icl: bool,
) -> list[VoiceClonePromptItem]:
    if use_icl and not ref_text:
        raise ValueError("ref_text is required when use_icl=True")

    return [
        VoiceClonePromptItem(
            ref_code=artifacts.ref_code if use_icl else None,
            ref_spk_embedding=artifacts.ref_spk_embedding,
            x_vector_only_mode=not use_icl,
            icl_mode=use_icl,
            ref_text=ref_text if use_icl else None,
        )
    ]


def _prepare_voice_clone_prompt(
    model: Qwen3TTSModel,
    *,
    ref_audio: object,
    ref_text: str | None,
    use_icl: bool,
) -> list[VoiceClonePromptItem]:
    cache_key = _voice_artifact_cache_key(
        ref_audio,
        ref_text=ref_text,
        use_icl=use_icl,
    )
    with _voice_artifact_cache_lock:
        cached = _voice_artifact_cache.get(cache_key)
        if cached is not None:
            _voice_artifact_cache.move_to_end(cache_key)
            logger.info(
                "voice_artifact_cache hit speaker=%s compute_ms=0.00",
                cache_key[:16],
            )
            return _build_voice_clone_prompt(
                cached,
                ref_text=ref_text,
                use_icl=use_icl,
            )

    started_at = perf_counter()
    artifacts = _compute_voice_artifacts(model, ref_audio)
    compute_ms = (perf_counter() - started_at) * 1000

    with _voice_artifact_cache_lock:
        cached = _voice_artifact_cache.get(cache_key)
        if cached is None:
            _voice_artifact_cache[cache_key] = artifacts
            _voice_artifact_cache.move_to_end(cache_key)
            while len(_voice_artifact_cache) > _voice_cache_capacity():
                _voice_artifact_cache.popitem(last=False)
            cached = artifacts
            logger.info(
                "voice_artifact_cache miss speaker=%s compute_ms=%.2f",
                cache_key[:16],
                compute_ms,
            )
        else:
            _voice_artifact_cache.move_to_end(cache_key)
            logger.info(
                "voice_artifact_cache hit speaker=%s compute_ms=0.00",
                cache_key[:16],
            )

    return _build_voice_clone_prompt(
        cached,
        ref_text=ref_text,
        use_icl=use_icl,
    )


def _patch_vocoder_chunk_size() -> None:
    """Honor RUST_TTS_VOCODER_CHUNK / QWEN_TTS_VOCODER_CHUNK on the 12Hz
    tokenizer's chunked_decode (default is 300). Smaller chunk = more
    left-context overlap = slower; larger = fewer forward calls but more
    memory per call."""
    raw = os.environ.get("QWEN_TTS_VOCODER_CHUNK") or os.environ.get(
        "RUST_TTS_VOCODER_CHUNK"
    )
    if not raw:
        return
    try:
        chunk_size = int(raw)
    except ValueError:
        logger.warning("invalid vocoder chunk size: %r", raw)
        return
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )

    original = Qwen3TTSTokenizerV2Decoder.chunked_decode

    def patched(self, codes, chunk_size=chunk_size, left_context_size=25):  # type: ignore[override]
        return original(self, codes, chunk_size, left_context_size)

    Qwen3TTSTokenizerV2Decoder.chunked_decode = patched
    logger.info("Patched vocoder chunked_decode chunk_size=%d", chunk_size)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _runtime_config

    _runtime_config = SpeechRuntimeConfig()
    _patch_vocoder_chunk_size()
    _clear_voice_artifact_cache()
    app.state.voice_artifact_cache = _voice_artifact_cache
    app.state.voice_artifact_cache_lock = _voice_artifact_cache_lock
    logger.info(
        "Voice artifact cache enabled: capacity=%d",
        _voice_cache_capacity(),
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
            model_path,
            device,
            dtype_name,
            attn_impl,
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
            base_model_path,
            device,
            dtype_name,
            attn_impl,
        )
        try:
            app.state.base_model = Qwen3TTSModel.from_pretrained(".", **kwargs)
            logger.info("Base model loaded successfully")
        finally:
            os.chdir(old_cwd)

    # Optional torch.compile for base/custom models. Off by default because
    # the first forward pays a multi-minute compile, and dynamic shapes can
    # trigger recompiles. Enable with QWEN_TTS_COMPILE=1 once you've
    # verified the workload is stable.
    compile_flag = os.environ.get("QWEN_TTS_COMPILE", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    compile_mode = os.environ.get("QWEN_TTS_COMPILE_MODE", "default")
    if compile_flag:
        # HF generation enters `Qwen3TTSForConditionalGeneration.talker.generate()`
        # and then repeatedly calls the deep transformer backbones on
        # `talker.model` and `talker.code_predictor.model`. Compiling the outer
        # wrapper is too shallow because GenerationMixin bypasses it via direct
        # attribute access on these inner modules.
        def _resolve_attr_path(root: object, attr_path: str) -> object | None:
            node = root
            for attr in attr_path.split("."):
                node = getattr(node, attr, None)
                if node is None:
                    return None
            return node

        def _compile_attr_path(root: object, root_name: str, attr_path: str) -> bool:
            parent_path, attr_name = attr_path.rsplit(".", 1)
            parent = _resolve_attr_path(root, parent_path)
            target = getattr(parent, attr_name, None) if parent else None
            full_path = f"{root_name}.{attr_path}"
            if target is None:
                logger.info(
                    "Skipping torch.compile for %s: attribute path not found",
                    full_path,
                )
                return False
            try:
                logger.info(
                    "Compiling %s with torch.compile(mode=%s, dynamic=True)",
                    full_path,
                    compile_mode,
                )
                setattr(
                    parent,
                    attr_name,
                    torch.compile(target, mode=compile_mode, dynamic=True),
                )
                logger.info("%s compiled", full_path)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("torch.compile on %s failed: %s", full_path, exc)
                return False

        compiled_attr_paths = (
            "model.talker.model",
            "model.talker.code_predictor.model",
        )
        for name in ("model", "base_model"):
            wrapper = getattr(app.state, name, None)
            if wrapper is None:
                continue
            compiled_any = False
            for attr_path in compiled_attr_paths:
                compiled_any = (
                    _compile_attr_path(wrapper, name, attr_path) or compiled_any
                )
            if not compiled_any:
                logger.warning(
                    "QWEN_TTS_COMPILE=1 but no deep backbone was compiled for %s",
                    name,
                )

    app.state.asr_model = None
    if asr_model_path:
        logger.info(
            "Loading ASR model %s on %s (%s, attn=%s)",
            asr_model_path,
            device,
            dtype_name,
            attn_impl,
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
    voice_clone_prompt: list[VoiceClonePromptItem] | None = None,
    ref_audio: object = None,
    ref_text: str | None = None,
    use_icl: bool = False,
    speaker: str = "",
    instruct: str = "",
) -> tuple[np.ndarray, int]:
    """Generate speech for a single text segment."""
    if is_voice_clone:
        generate_kwargs: dict[str, object] = {
            "text": text,
            "language": language,
        }
        if voice_clone_prompt is not None:
            generate_kwargs["voice_clone_prompt"] = voice_clone_prompt
        else:
            generate_kwargs.update(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=not use_icl,
            )
        wavs, sr = model.generate_voice_clone(**generate_kwargs)
    else:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
    return wavs[0], sr


def _generate_segment_talker_result(
    plan: SpeechSynthesisPlan,
    text: str,
) -> SegmentTalkerResult:
    """Generate codec tokens for one segment without running the vocoder yet."""
    model = plan.model

    if plan.is_voice_clone:
        if plan.voice_clone_prompt is None:
            raise ValueError("voice_clone_prompt must be prepared before pipelining")

        prompt_items = plan.voice_clone_prompt
        prompt_item = prompt_items[0]
        voice_clone_prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
        input_ids = model._tokenize_texts([model._build_assistant_text(text)])

        ref_ids: list[torch.Tensor | None] | None = None
        if prompt_item.ref_text:
            ref_ids = [
                model._tokenize_texts([model._build_ref_text(prompt_item.ref_text)])[0]
            ]

        gen_kwargs = model._merge_generate_kwargs()
        with _talker_lock:
            talker_codes_list, _ = model.model.generate(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt,
                languages=[plan.language],
                non_streaming_mode=False,
                **gen_kwargs,
            )

        codes = talker_codes_list[0]
        ref_code_length = 0
        if prompt_item.ref_code is not None:
            ref_code = prompt_item.ref_code.to(codes.device)
            ref_code_length = int(ref_code.shape[0])
            codes = torch.cat([ref_code, codes], dim=0)
        return SegmentTalkerResult(
            codes_for_decode=codes,
            ref_code_length=ref_code_length,
        )

    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    instruct_ids: list[torch.Tensor | None] = []
    if plan.instruct:
        instruct_ids.append(
            model._tokenize_texts([model._build_instruct_text(plan.instruct)])[0]
        )
    else:
        instruct_ids.append(None)

    gen_kwargs = model._merge_generate_kwargs()
    with _talker_lock:
        talker_codes_list, _ = model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[plan.language],
            speakers=[plan.speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )

    return SegmentTalkerResult(codes_for_decode=talker_codes_list[0])


def _decode_segment_audio(
    model: Qwen3TTSModel,
    talker_result: SegmentTalkerResult,
) -> tuple[np.ndarray, int]:
    """Decode codec tokens into waveform, matching the upstream wrapper behavior."""
    with _vocoder_lock:
        wavs, sample_rate = model.model.speech_tokenizer.decode(
            [{"audio_codes": talker_result.codes_for_decode}]
        )

    wav = wavs[0]
    if talker_result.ref_code_length > 0:
        total_len = int(talker_result.codes_for_decode.shape[0])
        cut = int(
            talker_result.ref_code_length / max(total_len, 1) * wav.shape[0]
        )
        wav = wav[cut:]
    return wav, sample_rate


def _submit_segment_talker_result(
    plan: SpeechSynthesisPlan,
    text: str,
) -> Future[SegmentTalkerResult]:
    return _segment_pipeline_executor.submit(
        _generate_segment_talker_result,
        plan,
        text,
    )


async def _parse_speech_request(raw_request: Request) -> ParsedSpeechRequest:
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
            ref_audio = _decode_audio_sample(audio_bytes, target_sr=TTS_SAMPLE_RATE)
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

    default_ref_path = os.environ.get("DEFAULT_AUDIO_SAMPLE_PATH", "")
    default_ref_text = os.environ.get("DEFAULT_AUDIO_SAMPLE_TEXT", "")
    if ref_audio is None and default_ref_path and os.path.isfile(default_ref_path):
        ref_audio = default_ref_path
        if not audio_sample_text and default_ref_text:
            audio_sample_text = default_ref_text

    return ParsedSpeechRequest(
        input_text=input_text,
        voice=voice,
        response_format=fmt,
        speed=speed,
        language=language,
        instructions=instructions,
        audio_sample_text=audio_sample_text,
        ref_audio=ref_audio,
    )


def _prepare_speech_plan(request: ParsedSpeechRequest) -> SpeechSynthesisPlan:
    assert _runtime_config is not None

    segments = split_text_for_tts(
        request.input_text,
        _runtime_config.segment_max_bytes,
        _runtime_config.segment_max_chars(),
        _runtime_config.segment_break_on_commas,
    )

    task_id = _task_control.alloc_task_id()
    _task_control.set_current(task_id, len(segments))
    try:
        logger.info(
            "speech task %d accepted: input_chars=%d, segments=%d",
            task_id,
            len(request.input_text),
            len(segments),
        )

        if request.ref_audio is not None:
            base_model: Qwen3TTSModel | None = app.state.base_model
            if base_model is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "audio_sample requires a base model. "
                        "Set TTS_BASE_MODEL_PATH to enable voice cloning."
                    ),
                )
            return SpeechSynthesisPlan(
                task_id=task_id,
                segments=segments,
                speed=request.speed,
                response_format=request.response_format,
                model=base_model,
                is_voice_clone=True,
                language=request.language,
                ref_audio=request.ref_audio,
                ref_text=request.audio_sample_text,
                use_icl=request.audio_sample_text is not None,
            )

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
        return SpeechSynthesisPlan(
            task_id=task_id,
            segments=segments,
            speed=request.speed,
            response_format=request.response_format,
            model=model,
            is_voice_clone=False,
            language=request.language,
            speaker=resolve_voice(request.voice),
            instruct=request.instructions or "",
        )
    except Exception:
        _task_control.clear_current(task_id)
        raise


async def _synthesize_audio_segments(
    plan: SpeechSynthesisPlan,
    raw_request: Request | None = None,
) -> AsyncIterator[tuple[np.ndarray, int]]:
    assert _runtime_config is not None
    emitted_audio = False
    pending_talker_result: Future[SegmentTalkerResult] | None = None

    try:
        if plan.is_voice_clone and plan.voice_clone_prompt is None:
            if raw_request is not None and await raw_request.is_disconnected():
                _task_control.cancel_task(plan.task_id)
            if _task_control.is_cancelled(plan.task_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"speech task {plan.task_id} was cancelled",
                )
            with _talker_lock:
                plan.voice_clone_prompt = _prepare_voice_clone_prompt(
                    plan.model,
                    ref_audio=plan.ref_audio,
                    ref_text=plan.ref_text,
                    use_icl=plan.use_icl,
                )

        for idx, segment in enumerate(plan.segments):
            if raw_request is not None and await raw_request.is_disconnected():
                _task_control.cancel_task(plan.task_id)

            if _task_control.is_cancelled(plan.task_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"speech task {plan.task_id} was cancelled",
                )

            logger.info(
                "speech task %d segment %d/%d: chars=%d",
                plan.task_id,
                idx + 1,
                len(plan.segments),
                len(segment),
            )

            if pending_talker_result is None:
                talker_result = _generate_segment_talker_result(plan, segment)
            else:
                talker_result = await asyncio.wrap_future(pending_talker_result)
                pending_talker_result = None

            next_idx = idx + 1
            if next_idx < len(plan.segments):
                pending_talker_result = _submit_segment_talker_result(
                    plan,
                    plan.segments[next_idx],
                )

            audio, sr = _decode_segment_audio(plan.model, talker_result)
            chunk = audio.astype(np.float32)

            if emitted_audio:
                gap_samples = int(sr * _runtime_config.segment_gap_seconds)
                gap = np.zeros(gap_samples, dtype=np.float32)
                chunk = np.concatenate((gap, chunk))

            emitted_audio = True
            _task_control.set_completed_segments(plan.task_id, idx + 1)
            yield chunk, sr
    except asyncio.CancelledError:
        _task_control.cancel_task(plan.task_id)
        raise
    finally:
        _task_control.clear_current(plan.task_id)


async def _collect_synthesized_audio(
    plan: SpeechSynthesisPlan,
    raw_request: Request | None = None,
) -> tuple[np.ndarray, int]:
    sample_rate = TTS_SAMPLE_RATE
    merged: list[np.ndarray] = []

    async for chunk, sr in _synthesize_audio_segments(plan, raw_request):
        sample_rate = sr
        merged.append(chunk)

    if not merged:
        return np.zeros(sample_rate * 2, dtype=np.float32), sample_rate

    return np.concatenate(merged), sample_rate


async def _stream_pcm_segments(
    plan: SpeechSynthesisPlan,
    raw_request: Request,
) -> AsyncGenerator[tuple[bytes, int, int], None]:
    async for chunk, sr in _synthesize_audio_segments(plan, raw_request):
        if plan.speed != 1.0:
            chunk = apply_speed(chunk, plan.speed)
        yield _encode_pcm(chunk), sr, _audio_channels(chunk)


@app.post("/v1/audio/speech")
async def create_speech(raw_request: Request) -> Response:
    """Generate audio from text (OpenAI-compatible).

    Accepts JSON or multipart/form-data. Automatically segments long text.
    """
    request = await _parse_speech_request(raw_request)
    plan = _prepare_speech_plan(request)
    audio, sr = await _collect_synthesized_audio(plan, raw_request)
    audio = apply_speed(audio, plan.speed)
    data = encode_audio(audio, sr, plan.response_format)
    return Response(content=data, media_type=CONTENT_TYPES[plan.response_format])


@app.post("/v1/audio/speech/stream")
async def create_streaming_speech(raw_request: Request) -> StreamingResponse:
    """Generate streaming WAV audio from text (OpenAI-compatible)."""
    request = await _parse_speech_request(raw_request)
    plan = _prepare_speech_plan(request)
    pcm_stream = _stream_pcm_segments(plan, raw_request)
    sample_rate = TTS_SAMPLE_RATE
    channels = 1
    first_chunk = b""

    try:
        first_chunk, sample_rate, channels = await anext(pcm_stream)
    except StopAsyncIteration:
        pass
    except Exception:
        await pcm_stream.aclose()
        raise

    async def stream_chunks() -> AsyncGenerator[bytes, None]:
        try:
            yield _wav_header(sample_rate, channels, WAV_STREAMING_DATA_SIZE)
            if first_chunk:
                yield first_chunk
            async for chunk, _, _ in pcm_stream:
                if chunk:
                    yield chunk
        except asyncio.CancelledError:
            _task_control.cancel_task(plan.task_id)
            raise
        finally:
            await pcm_stream.aclose()

    return StreamingResponse(
        stream_chunks(),
        media_type="audio/wav",
        headers={"Transfer-Encoding": "chunked"},
    )


# ---------------------------------------------------------------------------
# Cancel / status endpoints (Rust-client compatible)
# ---------------------------------------------------------------------------


@app.post("/v1/audio/speech/cancel-current")
async def cancel_current() -> JSONResponse:
    tid = _task_control.cancel_current()
    return JSONResponse(
        {
            "ok": True,
            "message": (
                f"Cancellation requested for current speech task {tid}"
                if tid
                else "No running speech task to cancel"
            ),
            "current_task_id": _task_control.status()["current_task_id"],
            "cancelled_up_to_task_id": tid or 0,
        }
    )


@app.post("/v1/audio/speech/cancel-all")
async def cancel_all() -> JSONResponse:
    cancelled_up_to = _task_control.cancel_all()
    return JSONResponse(
        {
            "ok": True,
            "message": (
                f"Cancellation requested for all speech tasks up to {cancelled_up_to}"
            ),
            "current_task_id": _task_control.status()["current_task_id"],
            "cancelled_up_to_task_id": cancelled_up_to,
        }
    )


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

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                src_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                wav_path,
            ],
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as ffmpeg_exc:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        os.unlink(src_path)
        raise HTTPException(status_code=400, detail="ffmpeg timeout") from ffmpeg_exc
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
