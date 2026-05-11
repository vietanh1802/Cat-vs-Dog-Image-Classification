"""
Deep-learning utilities for end-to-end transfer learning.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from modules.backbones import build_transfer_model
from modules.datasets import create_image_dataloaders
from modules.transforms import get_dl_transform


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _as_device(device: str | torch.device | None = None) -> torch.device:
    """Return a torch.device from a string/device/None value."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    """Validate and return an integer hyperparameter."""
    int_value = int(value)
    if allow_zero:
        if int_value < 0:
            raise ValueError(f"{name} must be non-negative, got {int_value}.")
    elif int_value <= 0:
        raise ValueError(f"{name} must be positive, got {int_value}.")
    return int_value


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable parameters and fail early if none are trainable."""
    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters found. Check model freezing/unfreezing.")
    return parameters


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone model state_dict to CPU for safe best-state restoration."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when DataParallel/DistributedDataParallel is used."""
    return model.module if hasattr(model, "module") else model


def _parameterized_children(model: nn.Module) -> list[nn.Module]:
    """Return direct child modules that own parameters."""
    core_model = _unwrap_model(model)
    return [
        child
        for child in core_model.children()
        if any(param.requires_grad is not None for param in child.parameters(recurse=True))
    ]


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 1) -> nn.Module:
    """Unfreeze the last parameterized child modules of a model.

    This is a generic fallback for torchvision models. For ResNet-like models,
    using ``num_blocks=2`` usually unfreezes the classifier head and the last
    convolutional block. For EfficientNet/VGG, it unfreezes the last direct
    parameterized children.
    """
    num_blocks = _validate_positive_int(num_blocks, "num_blocks", allow_zero=True)
    if num_blocks == 0:
        return model

    children = _parameterized_children(model)
    if not children:
        return model

    for child in children[-num_blocks:]:
        for param in child.parameters(recurse=True):
            param.requires_grad = True

    return model


def _compute_metrics(
    loss_sum: float,
    n_samples: int,
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    """Compute epoch metrics from accumulated predictions."""
    if n_samples == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "f1_macro": float("nan")}

    return {
        "loss": float(loss_sum / n_samples),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _scheduler_step(scheduler: Any | None, val_metrics: Mapping[str, float]) -> None:
    """Step scheduler while supporting ReduceLROnPlateau."""
    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_metrics.get("loss", np.inf))
    else:
        scheduler.step()


# -----------------------------------------------------------------------------
# Dataloaders and model construction
# -----------------------------------------------------------------------------


def create_image_dataloaders_for_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, torch.utils.data.DataLoader]:
    """Create train/validation/test dataloaders for a deep-learning config."""
    image_size = config.get("image_size", 224)
    batch_size = _validate_positive_int(config.get("batch_size", 32), "batch_size")
    num_workers = _validate_positive_int(config.get("num_workers", 0), "num_workers", allow_zero=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    transforms = {
        "train": get_dl_transform(image_size=image_size, train=True),
        "val": get_dl_transform(image_size=image_size, train=False),
        "test": get_dl_transform(image_size=image_size, train=False),
    }

    return create_image_dataloaders(
        splits=splits,
        transforms=transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(config.get("pin_memory", torch.cuda.is_available())),
        path_col=str(config.get("path_col", "path")),
        label_col=str(config.get("label_col", "label")),
    )


def build_model_for_config(
    config: Mapping[str, Any],
    num_classes: int,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Build a transfer-learning model from a deep-learning config."""
    device = _as_device(device or config.get("device", "auto"))

    model = build_transfer_model(
        backbone_name=str(config.get("backbone", "efficientnet_b0")),
        num_classes=num_classes,
        pretrained=bool(config.get("pretrained", True)),
        dropout=float(config.get("dropout", 0.2)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
        device=device,
    )

    return model


# -----------------------------------------------------------------------------
# Optimizer and scheduler construction
# -----------------------------------------------------------------------------


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer over trainable parameters only."""
    name = str(config.get("name", "adam")).lower().strip()
    lr = float(config.get("lr", 1e-3))

    if lr <= 0:
        raise ValueError(f"Optimizer learning rate must be positive, got {lr}.")

    parameters = _trainable_parameters(model)

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )

    raise ValueError("Unsupported optimizer name. Expected one of: 'sgd', 'adam', 'adamw'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any] | None,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build an optional learning-rate scheduler."""
    if not config:
        return None

    name = str(config.get("name", "none")).lower().strip()
    if name in {"", "none", "disabled", "off"}:
        return None

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=_validate_positive_int(config.get("step_size", 3), "step_size"),
            gamma=float(config.get("gamma", 0.1)),
        )

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=_validate_positive_int(config.get("t_max", 10), "t_max"),
        )

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "min")),
            factor=float(config.get("factor", 0.1)),
            patience=_validate_positive_int(config.get("patience", 2), "patience", allow_zero=True),
        )

    raise ValueError("Unsupported scheduler name. Expected one of: 'step', 'cosine', 'plateau', 'none'.")


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> dict[str, float]:
    """Train a model for one epoch and return loss/accuracy/F1."""
    device = _as_device(device)
    model.train()

    loss_sum = 0.0
    n_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        loss_sum += float(loss.item()) * batch_size
        n_samples += batch_size

        predictions = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    return _compute_metrics(loss_sum, n_samples, y_true, y_pred)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict[str, float]:
    """Evaluate a model for one epoch and return loss/accuracy/F1."""
    device = _as_device(device)
    model.eval()

    loss_sum = 0.0
    n_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = int(labels.size(0))
        loss_sum += float(loss.item()) * batch_size
        n_samples += batch_size

        predictions = outputs.argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    return _compute_metrics(loss_sum, n_samples, y_true, y_pred)


# -----------------------------------------------------------------------------
# Full fitting logic
# -----------------------------------------------------------------------------


def fit_transfer_model(
    model: nn.Module,
    dataloaders: Mapping[str, torch.utils.data.DataLoader],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit a transfer-learning model using head-training and optional fine-tuning.

    Expected config keys:
    - ``head_epochs`` or ``epochs``,
    - ``fine_tune_epochs`` optional,
    - ``head_lr`` and ``fine_tune_lr`` optional,
    - ``optimizer`` optional,
    - ``scheduler`` optional,
    - ``unfreeze_last_blocks`` optional.
    """
    if "train" not in dataloaders:
        raise KeyError("dataloaders must contain a 'train' loader.")
    if "val" not in dataloaders:
        raise KeyError("dataloaders must contain a 'val' loader.")

    device = _as_device(config.get("device", "auto"))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float | int | str]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -np.inf
    global_epoch = 0

    head_epochs = int(config.get("head_epochs", config.get("epochs", 1)))
    fine_tune_epochs = int(config.get("fine_tune_epochs", 0))

    if head_epochs < 0 or fine_tune_epochs < 0:
        raise ValueError("head_epochs and fine_tune_epochs must be non-negative.")

    phases: list[dict[str, Any]] = []
    if head_epochs > 0:
        phases.append({
            "name": "head",
            "epochs": head_epochs,
            "lr": float(config.get("head_lr", config.get("lr", 1e-3))),
            "unfreeze": False,
        })

    if fine_tune_epochs > 0:
        phases.append({
            "name": "fine_tune",
            "epochs": fine_tune_epochs,
            "lr": float(config.get("fine_tune_lr", config.get("lr", 1e-5))),
            "unfreeze": True,
        })

    if not phases:
        raise ValueError("At least one training epoch is required.")

    for phase in phases:
        if phase["unfreeze"]:
            unfreeze_last_blocks(
                model,
                num_blocks=int(config.get("unfreeze_last_blocks", 1)),
            )

        optimizer_config = dict(config.get("optimizer", {}))
        optimizer_config.setdefault("name", "adam")
        optimizer_config["lr"] = phase["lr"]

        optimizer = build_optimizer(model, optimizer_config)
        scheduler = build_scheduler(optimizer, config.get("scheduler", {}))

        for _ in range(int(phase["epochs"])):
            global_epoch += 1

            train_metrics = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            val_metrics = evaluate_one_epoch(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
            )

            row = {
                "epoch": global_epoch,
                "phase": phase["name"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)

            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = float(val_metrics["f1_macro"])
                best_state = _clone_state_dict_to_cpu(model)

            _scheduler_step(scheduler, val_metrics)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_val_f1_macro": float(best_val_f1),
    }


@torch.no_grad()
def predict_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict labels and probabilities for a dataloader."""
    device = _as_device(device)
    model = model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = probabilities.argmax(dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())
        y_prob.extend(probabilities.detach().cpu().tolist())

    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


# -----------------------------------------------------------------------------
# Checkpointing and model inspection
# -----------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Save model checkpoint, creating the parent directory when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load model checkpoint into a model and optionally an optimizer."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint


def count_trainable_parameters(model: nn.Module) -> dict[str, int | float]:
    """Return total/trainable/frozen parameter counts and trainable percentage."""
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": round(trainable / total * 100, 2) if total else 0.0,
    }


__all__ = [
    "create_image_dataloaders_for_config",
    "build_model_for_config",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate_one_epoch",
    "fit_transfer_model",
    "predict_dataloader",
    "save_checkpoint",
    "load_checkpoint",
    "count_trainable_parameters",
    "unfreeze_last_blocks",
]
