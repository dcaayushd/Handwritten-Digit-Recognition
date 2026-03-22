from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def build_versioned_model_path(models_dir: Path, version: str) -> Path:
    return models_dir / f"mnist_cnn_{version}.pt"


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    models_dir: Path
    model_paths: tuple[Path, ...]
    model_path: Path
    model_version: str
    log_level: str
    min_prediction_confidence: float
    min_prediction_margin: float
    input_size: int = 28
    mean: float = 0.1307
    std: float = 0.3081
    target_accuracy: float = 0.98
    seed: int = 42
    num_classes: int = 10
    allowed_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    allowed_content_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/jpg")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model_version = os.getenv("MODEL_VERSION", "v1").strip() or "v1"
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data"

    raw_model_path = os.getenv("MODEL_PATH", "").strip()
    requested_paths = [value.strip() for value in raw_model_path.split(",") if value.strip()]

    if requested_paths:
        model_paths = tuple(Path(value).expanduser() for value in requested_paths)
    else:
        model_paths = (build_versioned_model_path(models_dir, model_version),)

    normalized_paths: list[Path] = []
    for model_path in model_paths:
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        normalized_paths.append(model_path)

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        models_dir=models_dir,
        model_paths=tuple(normalized_paths),
        model_path=normalized_paths[0],
        model_version=model_version,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        min_prediction_confidence=float(os.getenv("MIN_PREDICTION_CONFIDENCE", "0.0")),
        min_prediction_margin=float(os.getenv("MIN_PREDICTION_MARGIN", "0.0")),
    )
