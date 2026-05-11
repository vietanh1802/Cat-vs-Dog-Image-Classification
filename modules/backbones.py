from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from modules.config_utils import SUPPORTED_BACKBONES, validate_and_normalize


def _load_base_model(name: str, pretrained: bool = True) -> nn.Module:
    name = validate_and_normalize(name, SUPPORTED_BACKBONES, "backbone")

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)

    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        return models.vgg16(weights=weights)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return models.efficientnet_b0(weights=weights)

    raise ValueError(f"Unsupported backbone: {name}")


def _strip_classifier(model: nn.Module, name: str) -> nn.Module:
    name = validate_and_normalize(name, SUPPORTED_BACKBONES, "backbone")

    if name == "resnet18":
        return nn.Sequential(*list(model.children())[:-1])
    if name in {"vgg16", "efficientnet_b0"}:
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    raise ValueError(f"Unsupported backbone: {name}")


def get_backbone(
    name: str,
    device: str | torch.device | None = None,
    pretrained: bool = True,
    freeze: bool = True,
    data_parallel: bool = False,
) -> nn.Module:
    """Load a pretrained backbone and strip its classification head."""
    name = validate_and_normalize(name, SUPPORTED_BACKBONES, "backbone")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = _load_base_model(name, pretrained=pretrained)
    model = _strip_classifier(base_model, name)

    if freeze:
        freeze_model(model)

    model = model.to(device)

    if data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    return model


def get_feature_dim(backbone_name: str) -> int:
    """Return the embedding dimension for a supported backbone."""
    dims = {"resnet18": 512, "vgg16": 512, "efficientnet_b0": 1280}
    key = validate_and_normalize(backbone_name, SUPPORTED_BACKBONES, "backbone")
    return dims[key]


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def replace_classifier_head(
    model: nn.Module,
    backbone_name: str,
    num_classes: int,
    dropout: float = 0.2,
) -> nn.Module:
    """Replace the classification head on a full torchvision model."""
    name = validate_and_normalize(name, SUPPORTED_BACKBONES, "backbone")

    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")

    if name == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name == "vgg16":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def build_transfer_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = True,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Build a transfer-learning model with a new classification head."""
    name = validate_and_normalize(name, SUPPORTED_BACKBONES, "backbone")
    model = _load_base_model(name, pretrained=pretrained)

    if freeze_backbone:
        freeze_model(model)

    model = replace_classifier_head(
        model=model,
        backbone_name=name,
        num_classes=num_classes,
        dropout=dropout,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model.to(device)


__all__ = [
  "get_backbone",
  "get_feature_dim",
  "freeze_model",
  "unfreeze_model",
  "replace_classifier_head",
  "build_transfer_model"
]