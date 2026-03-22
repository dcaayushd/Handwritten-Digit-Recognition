from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.model import MnistCNN


@pytest.fixture
def sample_image_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_digit.png"


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    torch.manual_seed(7)
    model = MnistCNN()
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "version": "test",
        "test_accuracy": 1.0,
        "class_names": [str(index) for index in range(10)],
        "mean": 0.1307,
        "std": 0.3081,
        "input_size": 28,
    }
    model_path = tmp_path / "mnist_cnn_test.pt"
    torch.save(checkpoint, model_path)
    return model_path


@pytest.fixture
def configured_settings(monkeypatch: pytest.MonkeyPatch, temp_model_path: Path):
    monkeypatch.setenv("MODEL_PATH", str(temp_model_path))
    monkeypatch.setenv("MODEL_VERSION", "test")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("MIN_PREDICTION_CONFIDENCE", "0.0")
    monkeypatch.setenv("MIN_PREDICTION_MARGIN", "0.0")
    get_settings.cache_clear()
    settings = get_settings()
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def client(configured_settings):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
