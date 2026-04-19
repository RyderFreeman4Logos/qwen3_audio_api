from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import main as api_main


def test_vram_fraction_unset_skips_cuda_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_TTS_VRAM_FRACTION", raising=False)
    set_fraction = Mock()
    monkeypatch.setattr(
        api_main.torch.cuda,
        "set_per_process_memory_fraction",
        set_fraction,
    )

    api_main._apply_cuda_vram_fraction("cuda:0")

    set_fraction.assert_not_called()


def test_vram_fraction_valid_calls_cuda_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_TTS_VRAM_FRACTION", "0.5")
    set_fraction = Mock()
    monkeypatch.setattr(api_main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api_main, "_resolve_cuda_device_index", lambda device: 0)
    monkeypatch.setattr(
        api_main.torch.cuda,
        "set_per_process_memory_fraction",
        set_fraction,
    )
    monkeypatch.setattr(
        api_main.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=8 * 1024 * 1024 * 1024),
    )

    api_main._apply_cuda_vram_fraction("cuda:0")

    set_fraction.assert_called_once_with(0.5, 0)


def test_vram_fraction_above_one_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_TTS_VRAM_FRACTION", "1.5")

    with pytest.raises(ValueError, match="must be in \\(0,1\\]"):
        api_main._apply_cuda_vram_fraction("cuda:0")


def test_vram_fraction_invalid_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("QWEN_TTS_VRAM_FRACTION", "invalid")
    set_fraction = Mock()
    monkeypatch.setattr(
        api_main.torch.cuda,
        "set_per_process_memory_fraction",
        set_fraction,
    )

    with caplog.at_level("WARNING"):
        api_main._apply_cuda_vram_fraction("cuda:0")

    assert "not a float, ignored" in caplog.text
    set_fraction.assert_not_called()


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", True),
        ("0", False),
    ],
)
def test_vram_warmup_bool_parser(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("QWEN_TTS_VRAM_WARMUP", raw_value)

    assert api_main._vram_warmup_enabled() is expected
