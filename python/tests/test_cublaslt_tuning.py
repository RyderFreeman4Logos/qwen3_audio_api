import os
from pathlib import Path

import torch

import main as api_main


def test_configure_cublaslt_tuning_env_creates_parent_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    tuning_file = tmp_path / "cache" / "cublaslt-heur.bin"
    monkeypatch.setenv("CUBLASLT_TUNING_DATA_FILE", str(tuning_file))
    monkeypatch.delenv("CUBLASLT_TUNING_MODE", raising=False)

    resolved = api_main.configure_cublaslt_tuning_env()

    assert resolved == tuning_file
    assert os.environ["CUBLASLT_TUNING_DATA_FILE"] == str(tuning_file)
    assert os.environ["CUBLASLT_TUNING_MODE"] == "HEURISTIC"
    assert tuning_file.parent.is_dir()


def test_configure_cuda_math_backends_enables_tf32() -> None:
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        api_main._configure_cuda_math_backends()

        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
