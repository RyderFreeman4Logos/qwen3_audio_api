import asyncio
import threading

import numpy as np
import pytest
import torch

import main as api_main


class FakeSpeechTokenizer:
    def __init__(
        self,
        events: list[str],
        *,
        second_generate_started: threading.Event | None = None,
        cancel_task_id: int | None = None,
    ) -> None:
        self._events = events
        self._second_generate_started = second_generate_started
        self._cancel_task_id = cancel_task_id

    def decode(self, items: list[dict[str, torch.Tensor]]) -> tuple[list[np.ndarray], int]:
        codes = items[0]["audio_codes"].reshape(-1)
        code_value = int(codes[-1].item())
        self._events.append(f"decode-start:{code_value}")
        if code_value == 1 and self._second_generate_started is not None:
            assert self._second_generate_started.wait(1.0)
        if code_value == 1 and self._cancel_task_id is not None:
            assert api_main._task_control.cancel_task(self._cancel_task_id) is True
        self._events.append(f"decode-end:{code_value}")
        return [np.array([float(code_value)], dtype=np.float32)], 24000


class FakeInnerModel:
    def __init__(
        self,
        events: list[str],
        *,
        second_generate_started: threading.Event | None = None,
        cancel_task_id: int | None = None,
    ) -> None:
        self.speech_tokenizer = FakeSpeechTokenizer(
            events,
            second_generate_started=second_generate_started,
            cancel_task_id=cancel_task_id,
        )
        self._events = events
        self._second_generate_started = second_generate_started

    def generate(self, *, input_ids: list[torch.Tensor], **_: object) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        code_value = int(input_ids[0].reshape(-1)[0].item())
        self._events.append(f"generate-start:{code_value}")
        if code_value == 2 and self._second_generate_started is not None:
            self._second_generate_started.set()
        self._events.append(f"generate-end:{code_value}")
        return [torch.tensor([code_value], dtype=torch.int64)], [torch.tensor([code_value], dtype=torch.float32)]


class FakePipelineModel:
    def __init__(
        self,
        events: list[str],
        *,
        second_generate_started: threading.Event | None = None,
        cancel_task_id: int | None = None,
    ) -> None:
        self.model = FakeInnerModel(
            events,
            second_generate_started=second_generate_started,
            cancel_task_id=cancel_task_id,
        )

    def _build_assistant_text(self, text: str) -> str:
        return text

    def _build_instruct_text(self, instruct: str) -> str:
        return instruct

    def _tokenize_texts(self, texts: list[str]) -> list[torch.Tensor]:
        return [torch.tensor([[int(texts[0])]], dtype=torch.int64)]

    def _merge_generate_kwargs(self, **_: object) -> dict[str, object]:
        return {}


def _install_runtime_config(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config = api_main.SpeechRuntimeConfig()
    runtime_config.segment_gap_seconds = 0.0
    monkeypatch.setattr(api_main, "_runtime_config", runtime_config)


def _make_plan(task_id: int, model: FakePipelineModel) -> api_main.SpeechSynthesisPlan:
    return api_main.SpeechSynthesisPlan(
        task_id=task_id,
        segments=["1", "2"],
        speed=1.0,
        response_format=api_main.ResponseFormat.wav,
        model=model,
        is_voice_clone=False,
        language="Auto",
        speaker="Vivian",
    )


def test_segment_pipeline_preserves_output_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_runtime_config(monkeypatch)
    events: list[str] = []
    second_generate_started = threading.Event()
    task_id = 301
    plan = _make_plan(
        task_id,
        FakePipelineModel(
            events,
            second_generate_started=second_generate_started,
        ),
    )

    api_main._task_control.set_current(task_id, len(plan.segments))
    audio, sample_rate = asyncio.run(api_main._collect_synthesized_audio(plan))

    assert sample_rate == 24000
    assert np.array_equal(audio, np.array([1.0, 2.0], dtype=np.float32))
    assert events.index("generate-start:2") < events.index("decode-end:1")


def test_segment_pipeline_cancellation_stops_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime_config(monkeypatch)
    events: list[str] = []
    task_id = 302
    plan = _make_plan(
        task_id,
        FakePipelineModel(
            events,
            cancel_task_id=task_id,
        ),
    )

    api_main._task_control.set_current(task_id, len(plan.segments))

    async def run_pipeline() -> tuple[list[np.ndarray], api_main.HTTPException]:
        chunks: list[np.ndarray] = []
        with pytest.raises(api_main.HTTPException) as exc_info:
            async for chunk, _ in api_main._synthesize_audio_segments(plan):
                chunks.append(chunk.copy())
        return chunks, exc_info.value

    chunks, exc = asyncio.run(run_pipeline())

    assert [chunk.tolist() for chunk in chunks] == [[1.0]]
    assert "cancelled" in str(exc.detail)
    assert events[:4] == [
        "generate-start:1",
        "generate-end:1",
        "decode-start:1",
        "decode-end:1",
    ]
    assert "decode-start:2" not in events


def test_segment_pipeline_overlaps_next_talker_and_serializes_vocoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime_config(monkeypatch)
    events: list[str] = []
    second_generate_started = threading.Event()
    task_id = 303
    plan = _make_plan(
        task_id,
        FakePipelineModel(
            events,
            second_generate_started=second_generate_started,
        ),
    )

    api_main._task_control.set_current(task_id, len(plan.segments))
    asyncio.run(api_main._collect_synthesized_audio(plan))

    assert events.index("generate-start:2") < events.index("decode-end:1")
    assert events.index("decode-end:1") < events.index("decode-start:2")
