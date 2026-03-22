from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.config import get_settings
from app.predict import Predictor
from app.utils import preprocess_image_bytes


def test_predictor_returns_expected_payload(configured_settings, sample_image_path):
    predictor = Predictor.from_settings(configured_settings)
    prediction = predictor.predict_image_bytes(
        sample_image_path.read_bytes(),
        filename=sample_image_path.name,
        content_type="image/png",
    )

    assert prediction.digit == 0
    assert prediction.confidence == pytest.approx(0.1, rel=1e-4)
    assert prediction.to_response() == {
        "digit": 0,
        "confidence": pytest.approx(0.1, rel=1e-4),
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_prediction(client, sample_image_path):
    with sample_image_path.open("rb") as handle:
        response = client.post(
            "/predict",
            files={"file": ("sample_digit.png", handle, "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"digit", "confidence"}
    assert body["digit"] == 0
    assert body["confidence"] == pytest.approx(0.1, rel=1e-4)


def test_predict_endpoint_rejects_invalid_file_type(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


def test_predict_endpoint_rejects_corrupt_image(client):
    response = client.post(
        "/predict",
        files={"file": ("broken.png", b"not-a-valid-image", "image/png")},
    )

    assert response.status_code == 400
    assert "valid PNG or JPEG" in response.json()["detail"]


def test_predict_endpoint_rejects_multiple_digits(client):
    image = Image.new("L", (240, 120), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((40, 20, 40, 95), fill=0, width=18)
    draw.line((120, 20, 120, 95), fill=0, width=18)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("two_digits.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    assert "one centered digit" in response.json()["detail"]


def test_preprocess_handles_lined_paper_single_digit():
    image = Image.new("L", (300, 420), color=245)
    draw = ImageDraw.Draw(image)
    for y_position in [60, 130, 200, 270, 340]:
        draw.line((0, y_position, 300, y_position), fill=150, width=3)

    draw.line((90, 160, 170, 160), fill=20, width=18)
    draw.line((170, 160, 120, 290), fill=20, width=18)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    tensor = preprocess_image_bytes(
        buffer.getvalue(),
        mean=0.1307,
        std=0.3081,
        input_size=28,
    )

    assert tensor.shape == (1, 1, 28, 28)
    assert float(tensor.abs().sum()) > 0.0


def test_preprocess_handles_vertical_lined_zero():
    image = Image.new("L", (320, 520), color=240)
    draw = ImageDraw.Draw(image)
    for x_position in [40, 90, 140, 190, 240, 290]:
        draw.line((x_position, 0, x_position, 520), fill=130, width=3)

    draw.ellipse((155, 210, 205, 255), outline=15, width=9)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    tensor = preprocess_image_bytes(
        buffer.getvalue(),
        mean=0.1307,
        std=0.3081,
        input_size=28,
    )

    assert tensor.shape == (1, 1, 28, 28)
    assert float(tensor.abs().sum()) > 0.0


def test_preprocess_handles_horizontal_lined_eight():
    image = Image.new("L", (520, 320), color=240)
    draw = ImageDraw.Draw(image)
    for y_position in [40, 95, 150, 205, 260]:
        draw.line((0, y_position, 520, y_position), fill=130, width=3)

    draw.ellipse((250, 115, 300, 165), outline=15, width=8)
    draw.ellipse((250, 160, 300, 220), outline=15, width=8)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    tensor = preprocess_image_bytes(
        buffer.getvalue(),
        mean=0.1307,
        std=0.3081,
        input_size=28,
    )

    assert tensor.shape == (1, 1, 28, 28)
    assert float(tensor.abs().sum()) > 0.0


def test_preprocess_handles_thin_centered_one():
    image = Image.new("L", (420, 420), color=245)
    draw = ImageDraw.Draw(image)
    draw.line((210, 120, 210, 290), fill=15, width=6)
    draw.line((170, 290, 250, 290), fill=15, width=5)
    draw.line((210, 120, 180, 155), fill=15, width=4)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    tensor = preprocess_image_bytes(
        buffer.getvalue(),
        mean=0.1307,
        std=0.3081,
        input_size=28,
    )

    assert tensor.shape == (1, 1, 28, 28)
    assert float(tensor.abs().sum()) > 0.0


def test_settings_support_multiple_model_paths(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "ensemble")
    monkeypatch.setenv("MODEL_PATH", "models/mnist_cnn_v2.pt,models/mnist_cnn_v3.pt")
    get_settings.cache_clear()
    settings = get_settings()

    assert len(settings.model_paths) == 2
    assert settings.model_paths[0].name == "mnist_cnn_v2.pt"
    assert settings.model_paths[1].name == "mnist_cnn_v3.pt"

    get_settings.cache_clear()


def test_settings_resolve_versioned_model_path(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("MODEL_VERSION", "v9")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.model_path == settings.models_dir / "mnist_cnn_v9.pt"

    get_settings.cache_clear()


def test_predictor_rejects_uncertain_predictions(monkeypatch, temp_model_path):
    monkeypatch.setenv("MODEL_PATH", str(temp_model_path))
    monkeypatch.setenv("MODEL_VERSION", "test")
    monkeypatch.setenv("MIN_PREDICTION_CONFIDENCE", "0.7")
    monkeypatch.setenv("MIN_PREDICTION_MARGIN", "0.2")
    get_settings.cache_clear()
    settings = get_settings()
    predictor = Predictor.from_settings(settings)

    with pytest.raises(ValueError, match="uncertain"):
        predictor.predict_image_bytes(
            Path(__file__).parent.joinpath("fixtures", "sample_digit.png").read_bytes(),
            filename="sample_digit.png",
            content_type="image/png",
        )

    get_settings.cache_clear()
