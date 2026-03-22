from __future__ import annotations

import argparse
import copy
import logging
import random
import ssl
from dataclasses import dataclass
from pathlib import Path

import certifi
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from app.config import get_settings
from app.model import build_model, load_checkpoint
from app.utils import (
    InputValidationError,
    build_inference_batch,
    ensure_directory,
    resolve_model_path,
    select_device,
    set_global_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class CachedUserDigitSample:
    path: Path
    label: int
    views: tuple[torch.Tensor, ...]


class RandomPhotoArtifacts:
    """Augment MNIST tensors so the model sees notebook-style photos during training."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        augmented = tensor.clone()
        _, height, width = augmented.shape

        if random.random() < 0.80:
            y_axis = torch.linspace(-1.0, 1.0, height, dtype=augmented.dtype).view(1, height, 1)
            x_axis = torch.linspace(-1.0, 1.0, width, dtype=augmented.dtype).view(1, 1, width)
            gradient = (
                random.uniform(-0.14, 0.14) * x_axis
                + random.uniform(-0.14, 0.14) * y_axis
                + random.uniform(0.0, 0.10)
            )
            augmented = torch.clamp(augmented + gradient, 0.0, 1.0)

        if random.random() < 0.65:
            line_intensity = random.uniform(0.05, 0.20)
            thickness = random.randint(1, 2)
            spacing = random.randint(5, 9)
            offset = random.randint(0, spacing - 1)
            if random.random() < 0.5:
                for row in range(offset, height, spacing):
                    row_end = min(height, row + thickness)
                    augmented[:, row:row_end, :] = torch.maximum(
                        augmented[:, row:row_end, :],
                        torch.full_like(augmented[:, row:row_end, :], line_intensity),
                    )
            else:
                for col in range(offset, width, spacing):
                    col_end = min(width, col + thickness)
                    augmented[:, :, col:col_end] = torch.maximum(
                        augmented[:, :, col:col_end],
                        torch.full_like(augmented[:, :, col:col_end], line_intensity),
                    )

        if random.random() < 0.50:
            if random.random() < 0.5:
                augmented = F.max_pool2d(augmented.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
            else:
                augmented = -F.max_pool2d((-augmented).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
                augmented = torch.clamp(augmented, 0.0, 1.0)

        if random.random() < 0.55:
            noise = torch.randn_like(augmented) * random.uniform(0.01, 0.06)
            augmented = torch.clamp(augmented + noise, 0.0, 1.0)

        if random.random() < 0.35:
            augmented = torch.clamp(augmented * random.uniform(0.70, 1.20), 0.0, 1.0)

        return augmented


def configure_ssl_certificates() -> None:
    """Use certifi so torchvision dataset downloads work on more local Python installs."""

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST handwritten digit CNN.")
    parser.add_argument("--version", default="v3", help="Version suffix for the saved model artifact.")
    parser.add_argument("--architecture", default="mnist_resnet", help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1.5e-3, help="Peak learning rate.")
    parser.add_argument(
        "--fine-tune-from",
        default="",
        help="Optional checkpoint path to warm-start from before training.",
    )
    parser.add_argument(
        "--user-data-dir",
        default="",
        help="Optional directory with labeled personal digit photos in subfolders 0-9.",
    )
    parser.add_argument(
        "--user-oversample",
        type=int,
        default=10,
        help="How many times to repeat personal handwriting samples during training.",
    )
    return parser.parse_args()


class CachedPhotoDigitDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[CachedUserDigitSample], *, training: bool) -> None:
        self.samples = samples
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        if self.training and len(sample.views) > 1:
            view = sample.views[random.randrange(len(sample.views))]
        else:
            view = sample.views[0]
        return view.clone(), sample.label


def _load_user_digit_samples(
    *,
    root_dir: Path,
    mean: float,
    std: float,
    input_size: int,
) -> list[CachedUserDigitSample]:
    samples: list[CachedUserDigitSample] = []

    for label in range(10):
        label_dir = root_dir / str(label)
        if not label_dir.exists():
            continue

        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not image_path.is_file():
                continue

            try:
                batch = build_inference_batch(
                    image_path.read_bytes(),
                    mean=mean,
                    std=std,
                    input_size=input_size,
                )
            except InputValidationError as exc:
                logger.warning("Skipping invalid custom sample path=%s error=%s", image_path, exc)
                continue

            samples.append(
                CachedUserDigitSample(
                    path=image_path,
                    label=label,
                    views=tuple(batch[view_index].cpu() for view_index in range(batch.shape[0])),
                )
            )

    return samples


def _split_user_digit_samples(
    samples: list[CachedUserDigitSample],
    *,
    seed: int,
) -> tuple[list[CachedUserDigitSample], list[CachedUserDigitSample]]:
    train_samples: list[CachedUserDigitSample] = []
    val_samples: list[CachedUserDigitSample] = []
    random_generator = random.Random(seed)

    for label in range(10):
        label_samples = [sample for sample in samples if sample.label == label]
        random_generator.shuffle(label_samples)

        if len(label_samples) <= 1:
            train_samples.extend(label_samples)
            continue

        if len(label_samples) < 5:
            val_count = 1
        else:
            val_count = max(1, int(round(len(label_samples) * 0.2)))

        val_count = min(val_count, len(label_samples) - 1)
        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    return train_samples, val_samples


def create_dataloaders(
    *,
    data_dir: str,
    batch_size: int,
    seed: int,
    mean: float,
    std: float,
    input_size: int,
    pin_memory: bool,
    user_data_dir: str | None,
    user_oversample: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=18,
                translate=(0.18, 0.18),
                scale=(0.75, 1.25),
                shear=10,
                fill=0,
            ),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.35, fill=0),
            transforms.ToTensor(),
            RandomPhotoArtifacts(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.35,
            ),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )

    augmented_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    eval_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=eval_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(augmented_train), generator=generator).tolist()
    train_indices = indices[:55_000]
    val_indices = indices[55_000:]
    train_dataset: Dataset[tuple[torch.Tensor, int]] = Subset(augmented_train, train_indices)
    val_dataset: Dataset[tuple[torch.Tensor, int]] = Subset(eval_train, val_indices)

    if user_data_dir:
        user_root = Path(user_data_dir).expanduser()
        if not user_root.is_absolute():
            user_root = get_settings().project_root / user_root

        if user_root.exists():
            user_samples = _load_user_digit_samples(
                root_dir=user_root,
                mean=mean,
                std=std,
                input_size=input_size,
            )
            user_train_samples, user_val_samples = _split_user_digit_samples(user_samples, seed=seed)

            if user_train_samples:
                user_train_dataset = CachedPhotoDigitDataset(user_train_samples, training=True)
                repeated_user_datasets = [user_train_dataset for _ in range(max(1, user_oversample))]
                train_dataset = ConcatDataset([train_dataset, *repeated_user_datasets])
            if user_val_samples:
                user_val_dataset = CachedPhotoDigitDataset(user_val_samples, training=False)
                val_dataset = ConcatDataset([val_dataset, user_val_dataset])

            logger.info(
                "Loaded %s personal handwriting samples from %s (%s train / %s val, oversample=%s)",
                len(user_samples),
                user_root,
                len(user_train_samples),
                len(user_val_samples),
                max(1, user_oversample),
            )
        else:
            logger.warning("Personal handwriting directory was not found: %s", user_root)

    num_workers = min(4, max(1, (torch.get_num_threads() or 1) // 2))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def _resolve_optional_checkpoint(path_value: str, *, project_root: Path) -> Path | None:
    cleaned = path_value.strip()
    if not cleaned:
        return None
    checkpoint_path = Path(cleaned).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    return checkpoint_path


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: AdamW | None = None,
    scheduler: OneCycleLR | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    return EpochMetrics(
        loss=total_loss / total_items,
        accuracy=total_correct / total_items,
    )


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    version: str,
    architecture: str,
    test_accuracy: float,
    mean: float,
    std: float,
    input_size: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "version": version,
            "architecture": architecture,
            "test_accuracy": test_accuracy,
            "class_names": [str(index) for index in range(10)],
            "mean": mean,
            "std": std,
            "input_size": input_size,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_ssl_certificates()
    set_global_seed(settings.seed)
    ensure_directory(settings.data_dir)
    ensure_directory(settings.models_dir)

    device = select_device()
    logger.info("Using device=%s", device)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=str(settings.data_dir),
            batch_size=args.batch_size,
            seed=settings.seed,
            mean=settings.mean,
            std=settings.std,
            input_size=settings.input_size,
            pin_memory=device.type == "cuda",
            user_data_dir=args.user_data_dir or None,
            user_oversample=args.user_oversample,
        )
    except ModuleNotFoundError as exc:
        logger.error("torchvision is required for training: %s", exc)
        return 1

    model = build_model(args.architecture, num_classes=settings.num_classes).to(device)
    fine_tune_checkpoint = _resolve_optional_checkpoint(args.fine_tune_from, project_root=settings.project_root)
    if fine_tune_checkpoint is not None:
        if not fine_tune_checkpoint.exists():
            logger.error("Fine-tune checkpoint was not found: %s", fine_tune_checkpoint)
            return 1
        loaded_checkpoint = load_checkpoint(fine_tune_checkpoint, device)
        if loaded_checkpoint.architecture.strip().lower() != args.architecture.strip().lower():
            logger.error(
                "Checkpoint architecture '%s' does not match requested architecture '%s'.",
                loaded_checkpoint.architecture,
                args.architecture,
            )
            return 1
        model.load_state_dict(loaded_checkpoint.state_dict)
        logger.info(
            "Warm-started training from checkpoint=%s version=%s",
            fine_tune_checkpoint,
            loaded_checkpoint.version,
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate / 25, weight_decay=5e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.18,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=500.0,
    )

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_metrics = run_epoch(model, val_loader, criterion, device=device)

        logger.info(
            "Epoch %s/%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6f",
            epoch,
            args.epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
            optimizer.param_groups[0]["lr"],
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    test_metrics = run_epoch(model, test_loader, criterion, device=device)

    output_path = resolve_model_path(settings, version=args.version)
    save_checkpoint(
        path=str(output_path),
        model=model,
        version=args.version,
        architecture=args.architecture,
        test_accuracy=test_metrics.accuracy,
        mean=settings.mean,
        std=settings.std,
        input_size=settings.input_size,
    )

    logger.info(
        "Saved model version=%s architecture=%s path=%s test_loss=%.4f test_acc=%.4f",
        args.version,
        args.architecture,
        output_path,
        test_metrics.loss,
        test_metrics.accuracy,
    )

    if test_metrics.accuracy < settings.target_accuracy:
        logger.error(
            "Model accuracy %.4f did not reach target %.2f",
            test_metrics.accuracy,
            settings.target_accuracy,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
