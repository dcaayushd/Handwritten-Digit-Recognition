from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from app.config import Settings
from app.model import build_model, load_checkpoint
from app.utils import (
    InferenceCandidate,
    InputValidationError,
    build_inference_candidates,
    select_device,
    validate_upload,
)

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the configured model artifact cannot be loaded."""


@dataclass(frozen=True)
class Prediction:
    digit: int
    confidence: float

    def to_response(self) -> dict[str, float | int]:
        return asdict(self)


class Predictor:
    """Reusable inference service that keeps the model loaded in memory."""

    def __init__(
        self,
        *,
        models: list[nn.Module],
        device: torch.device,
        settings: Settings,
        mean: float,
        std: float,
        input_size: int,
        class_names: list[str],
        versions: list[str],
    ) -> None:
        self.models = models
        self.device = device
        self.settings = settings
        self.mean = mean
        self.std = std
        self.input_size = input_size
        self.class_names = class_names
        self.versions = versions

    @classmethod
    def from_settings(cls, settings: Settings) -> "Predictor":
        device = select_device()
        loaded_models: list[nn.Module] = []
        versions: list[str] = []
        reference_class_names: list[str] | None = None
        reference_mean: float | None = None
        reference_std: float | None = None
        reference_input_size: int | None = None

        for checkpoint_path in settings.model_paths:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise ModelLoadError(
                    f"Model checkpoint was not found at '{checkpoint_path}'. "
                    "Train the model first or point MODEL_PATH to an existing checkpoint."
                )

            try:
                checkpoint = load_checkpoint(checkpoint_path, device)
                model = build_model(checkpoint.architecture, num_classes=len(checkpoint.class_names))
                model.load_state_dict(checkpoint.state_dict)
            except Exception as exc:  # pragma: no cover - startup failures are asserted by behavior
                raise ModelLoadError(f"Failed to load model checkpoint '{checkpoint_path}': {exc}") from exc

            if reference_class_names is None:
                reference_class_names = checkpoint.class_names
                reference_mean = checkpoint.mean
                reference_std = checkpoint.std
                reference_input_size = checkpoint.input_size
            else:
                if checkpoint.class_names != reference_class_names:
                    raise ModelLoadError("Configured checkpoints use incompatible class names.")
                if checkpoint.input_size != reference_input_size:
                    raise ModelLoadError("Configured checkpoints use incompatible input sizes.")

            model.to(device)
            model.eval()
            loaded_models.append(model)
            versions.append(checkpoint.version)
            logger.info(
                "Loaded model version=%s architecture=%s from %s on device=%s",
                checkpoint.version,
                checkpoint.architecture,
                checkpoint_path,
                device,
            )

        return cls(
            models=loaded_models,
            device=device,
            settings=settings,
            mean=reference_mean or settings.mean,
            std=reference_std or settings.std,
            input_size=reference_input_size or settings.input_size,
            class_names=reference_class_names or [str(index) for index in range(settings.num_classes)],
            versions=versions,
        )

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str | None,
        content_type: str | None,
    ) -> Prediction:
        validate_upload(filename, content_type, self.settings)
        candidates = build_inference_candidates(
            image_bytes,
            mean=self.mean,
            std=self.std,
            input_size=self.input_size,
        )
        return self.predict_candidates(candidates)

    def predict_tensor(self, tensor: torch.Tensor) -> Prediction:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("Prediction tensor must have shape [batch, channels, height, width].")

        return self.predict_candidates(
            [
                InferenceCandidate(
                    tensor=tensor,
                    source="tensor_input",
                    hole_count=0,
                    hole_vertical_bias=0.0,
                    aspect_ratio=1.0,
                )
            ]
        )

    def _predict_candidate_probabilities(self, candidate: InferenceCandidate) -> torch.Tensor:
        with torch.inference_mode():
            batch = candidate.tensor.to(self.device)
            aggregated_probabilities = None
            for model in self.models:
                logits = model(batch)
                probabilities = torch.softmax(logits, dim=1).mean(dim=0).cpu()
                aggregated_probabilities = (
                    probabilities
                    if aggregated_probabilities is None
                    else aggregated_probabilities + probabilities
                )

        probabilities = aggregated_probabilities / len(self.models)
        return self._apply_structural_prior(probabilities, candidate)

    def _apply_structural_prior(self, probabilities: torch.Tensor, candidate: InferenceCandidate) -> torch.Tensor:
        if float(torch.max(probabilities).item()) <= 0.18:
            return probabilities

        adjusted = probabilities.clone()

        if candidate.hole_count == 0:
            adjusted[0] *= 0.20
            adjusted[8] *= 0.15
            adjusted[6] *= 0.65
            adjusted[9] *= 0.65
            if candidate.aspect_ratio <= 0.52:
                adjusted[1] *= 1.50
            if 0.45 <= candidate.aspect_ratio <= 0.95:
                adjusted[4] *= 1.15
                adjusted[7] *= 1.10
        elif candidate.hole_count >= 2:
            adjusted[8] *= 1.80
            adjusted[0] *= 0.60
            adjusted[6] *= 0.55
            adjusted[9] *= 0.55
        else:
            adjusted[1] *= 0.55
            adjusted[7] *= 0.80
            adjusted[8] *= 0.70
            if candidate.hole_vertical_bias >= 0.10:
                adjusted[6] *= 2.20
                adjusted[0] *= 0.38
                adjusted[9] *= 0.70
            elif candidate.hole_vertical_bias <= -0.10:
                adjusted[9] *= 2.20
                adjusted[0] *= 0.38
                adjusted[6] *= 0.70
            else:
                adjusted[0] *= 1.20
                adjusted[6] *= 0.92
                adjusted[9] *= 0.92

        total = float(adjusted.sum().item())
        if total <= 0.0:
            return probabilities
        return adjusted / total

    def predict_candidates(self, candidates: list[InferenceCandidate]) -> Prediction:
        if not candidates:
            raise ValueError("At least one inference candidate is required.")

        candidate_summaries: list[tuple[float, float, int, torch.Tensor, str, list[str]]] = []
        for candidate in candidates:
            probabilities = self._predict_candidate_probabilities(candidate)
            top_count = min(3, len(self.class_names))
            top_confidences, top_indices = torch.topk(probabilities, k=top_count)
            top_confidence = float(top_confidences[0].item())
            runner_up_confidence = float(top_confidences[1].item()) if top_count > 1 else 0.0
            margin = top_confidence - runner_up_confidence
            digit = int(top_indices[0].item())
            top_predictions = [
                f"{int(class_index)}:{float(class_confidence):.4f}"
                for class_confidence, class_index in zip(top_confidences.tolist(), top_indices.tolist(), strict=False)
            ]
            candidate_summaries.append(
                (
                    top_confidence + (margin * 0.35),
                    top_confidence,
                    digit,
                    probabilities,
                    candidate.source,
                    top_predictions,
                )
            )

        candidate_summaries.sort(key=lambda summary: summary[0], reverse=True)
        _, top_confidence, digit, probabilities, source, top_predictions = candidate_summaries[0]
        runner_up_confidence = 0.0
        if len(top_predictions) > 1:
            runner_up_confidence = float(
                sorted(probabilities.tolist(), reverse=True)[1]
            )
        margin = top_confidence - runner_up_confidence

        if (
            top_confidence < self.settings.min_prediction_confidence
            or margin < self.settings.min_prediction_margin
        ):
            logger.warning(
                "Prediction rejected as uncertain source=%s confidence=%.4f margin=%.4f thresholds=(%.4f, %.4f) top=%s candidates=%s",
                source,
                top_confidence,
                margin,
                self.settings.min_prediction_confidence,
                self.settings.min_prediction_margin,
                top_predictions,
                [
                    {
                        "source": candidate_source,
                        "digit": candidate_digit,
                        "score": round(candidate_score, 4),
                        "confidence": round(candidate_confidence, 4),
                        "top": candidate_top_predictions,
                    }
                    for candidate_score, candidate_confidence, candidate_digit, _, candidate_source, candidate_top_predictions in candidate_summaries
                ],
            )
            raise InputValidationError(
                "Prediction is uncertain. Upload a tighter crop or fine-tune with your handwriting samples."
            )

        prediction = Prediction(digit=digit, confidence=top_confidence)
        logger.info(
            "Prediction made digit=%s confidence=%.4f source=%s ensemble=%s top=%s",
            prediction.digit,
            prediction.confidence,
            source,
            ",".join(self.versions),
            top_predictions,
        )
        return prediction
