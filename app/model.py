from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class MnistCNN(nn.Module):
    """Compact CNN tuned for MNIST-sized handwritten digit inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.30),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.layers(inputs)
        return self.activation(outputs + residual)


class MnistResNet(nn.Module):
    """Residual CNN with batch norm for stronger robustness on photo-like inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.backbone(outputs)
        return self.head(outputs)


@dataclass(frozen=True)
class LoadedCheckpoint:
    state_dict: dict[str, Any]
    version: str
    architecture: str
    class_names: list[str]
    mean: float
    std: float
    input_size: int
    test_accuracy: float


def build_model(architecture: str, num_classes: int = 10) -> nn.Module:
    normalized = architecture.strip().lower()
    if normalized in {"mnist_cnn", "cnn"}:
        return MnistCNN(num_classes=num_classes)
    if normalized in {"mnist_resnet", "resnet"}:
        return MnistResNet(num_classes=num_classes)
    raise ValueError(f"Unsupported model architecture '{architecture}'.")


def load_checkpoint(checkpoint_path: Path | str, device: torch.device) -> LoadedCheckpoint:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    required_keys = {
        "model_state_dict",
        "version",
        "class_names",
        "mean",
        "std",
        "input_size",
        "test_accuracy",
    }
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    return LoadedCheckpoint(
        state_dict=checkpoint["model_state_dict"],
        version=str(checkpoint["version"]),
        architecture=str(checkpoint.get("architecture", "mnist_cnn")),
        class_names=[str(value) for value in checkpoint["class_names"]],
        mean=float(checkpoint["mean"]),
        std=float(checkpoint["std"]),
        input_size=int(checkpoint["input_size"]),
        test_accuracy=float(checkpoint["test_accuracy"]),
    )
