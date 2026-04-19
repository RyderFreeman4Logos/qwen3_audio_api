import os
from pathlib import Path


def _setdefault_cublaslt_tuning_env() -> Path:
    cache_path = Path(
        os.environ.setdefault(
            "CUBLASLT_TUNING_DATA_FILE",
            str(Path.home() / ".cache" / "qwen3-audio-api" / "cublaslt-heur.bin"),
        )
    ).expanduser()
    os.environ.setdefault("CUBLASLT_TUNING_MODE", "HEURISTIC")
    return cache_path


def configure_cublaslt_tuning_env() -> Path:
    cache_path = _setdefault_cublaslt_tuning_env()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "Unable to create cuBLASLt tuning cache directory "
            f"{cache_path.parent} for CUBLASLT_TUNING_DATA_FILE"
        ) from exc
    return cache_path


_setdefault_cublaslt_tuning_env()
