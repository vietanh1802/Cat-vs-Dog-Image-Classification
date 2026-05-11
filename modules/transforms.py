"""
Image preprocessing transforms for classical feature extraction and deep learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


SUPPORTED_TRANSFORM_MODES = (
    "stretch",
    "center_crop",
    "letterbox",
    "augmented",
)


def _validate_image_size(image_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Return image size as ``(height, width)`` after validation."""
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return image_size, image_size

    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError("image_size tuple/list must contain exactly two integers.")
        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size dimensions must be positive.")
        return height, width

    raise TypeError("image_size must be an int or a length-2 tuple/list.")


def _normalize_mode(mode: str) -> str:
    """Normalize and validate preprocessing mode."""
    normalized = str(mode).lower().strip()
    if normalized not in SUPPORTED_TRANSFORM_MODES:
        raise ValueError(
            f"Unsupported transform mode: {mode}. "
            f"Supported modes: {list(SUPPORTED_TRANSFORM_MODES)}."
        )
    return normalized


def _validate_fill(fill: int | tuple[int, int, int]) -> int | tuple[int, int, int]:
    """Validate a PIL-compatible fill value."""
    if isinstance(fill, int):
        if not 0 <= fill <= 255:
            raise ValueError("Integer fill must be in [0, 255].")
        return fill

    if isinstance(fill, tuple) and len(fill) == 3:
        values = tuple(int(value) for value in fill)
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("RGB fill values must be in [0, 255].")
        return values

    raise TypeError("fill must be an int or an RGB tuple of length 3.")


class LetterBox:
    """Resize an image while preserving aspect ratio, then pad to target size.

    Parameters
    ----------
    size:
        Target size. An integer creates a square output. A tuple/list is treated
        as ``(height, width)``.
    fill:
        Padding color.
    interpolation:
        PIL interpolation method used for resizing.
    """

    def __init__(
        self,
        size: int | tuple[int, int] | list[int],
        fill: int | tuple[int, int, int] = (0, 0, 0),
        interpolation: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> None:
        self.target_height, self.target_width = _validate_image_size(size)
        self.fill = _validate_fill(fill)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        width, height = img.size

        if width <= 0 or height <= 0:
            return Image.new("RGB", (self.target_width, self.target_height), self.fill)

        scale = min(self.target_width / width, self.target_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = img.resize((new_width, new_height), self.interpolation)
        canvas = Image.new("RGB", (self.target_width, self.target_height), self.fill)

        left = (self.target_width - new_width) // 2
        top = (self.target_height - new_height) // 2
        canvas.paste(resized, (left, top))

        return canvas


class SquarePad:
    """Pad an image to a square shape without resizing."""

    def __init__(self, fill: int | tuple[int, int, int] = (0, 0, 0)) -> None:
        self.fill = _validate_fill(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        width, height = img.size
        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_top = (max_side - height) // 2
        pad_right = max_side - width - pad_left
        pad_bottom = max_side - height - pad_top

        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def get_imagenet_mean_std() -> tuple[list[float], list[float]]:
    """Return ImageNet normalization mean and standard deviation."""
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_normalize_transform(normalize: str | bool = "imagenet") -> T.Normalize | T.Lambda:
    """Return a normalization transform.

    ``normalize`` may be:
    - ``"imagenet"`` or True: ImageNet normalization,
    - ``"none"`` or False: identity transform.
    """
    if normalize is True:
        normalize = "imagenet"
    if normalize is False or str(normalize).lower().strip() in {"none", "false", "off", "no"}:
        return T.Lambda(lambda tensor: tensor)

    normalized = str(normalize).lower().strip()
    if normalized != "imagenet":
        raise ValueError("Only normalize='imagenet' or normalize='none' are supported.")

    mean, std = get_imagenet_mean_std()
    return T.Normalize(mean=mean, std=std)


def get_hybrid_transform(
    mode: str,
    image_size: int | tuple[int, int] | list[int] = 224,
    train: bool = False,
    normalize: str | bool = "imagenet",
) -> T.Compose:
    """Return preprocessing transform for frozen-backbone feature extraction.

    Frozen feature extraction should usually be deterministic. The only mode that
    uses random augmentation is ``mode='augmented'`` with ``train=True``.
    """
    mode = _normalize_mode(mode)
    height, width = _validate_image_size(image_size)
    normalize_transform = get_normalize_transform(normalize)

    if mode == "stretch":
        return T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            normalize_transform,
        ])

    if mode == "center_crop":
        resize_size = int(round(max(height, width) * 1.14))
        return T.Compose([
            T.Resize(resize_size),
            T.CenterCrop((height, width)),
            T.ToTensor(),
            normalize_transform,
        ])

    if mode == "letterbox":
        return T.Compose([
            LetterBox((height, width)),
            T.ToTensor(),
            normalize_transform,
        ])

    # mode == "augmented"
    if train:
        return T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            normalize_transform,
        ])

    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize_transform,
    ])


def get_dl_transform(
    image_size: int | tuple[int, int] | list[int] = 224,
    train: bool = False,
    normalize: str | bool = "imagenet",
) -> T.Compose:
    """Return preprocessing transform for end-to-end deep learning."""
    return get_hybrid_transform(
        mode="augmented" if train else "stretch",
        image_size=image_size,
        train=train,
        normalize=normalize,
    )


def build_image_transform(
    config: dict[str, Any],
    split: str = "train",
) -> T.Compose:
    """Build an image transform from a preprocessing config.

    Expected config keys:
    - ``mode``: stretch, center_crop, letterbox, or augmented,
    - ``image_size``: int or (height, width),
    - ``normalize``: imagenet or none,
    - ``train_augmentation``: whether train split should use augmented mode.
    """
    split_name = str(split).lower().strip()
    mode = str(config.get("mode", "letterbox"))
    train = split_name == "train"

    if train and bool(config.get("train_augmentation", False)):
        mode = "augmented"

    return get_hybrid_transform(
        mode=mode,
        image_size=config.get("image_size", 224),
        train=train,
        normalize=config.get("normalize", "imagenet"),
    )


def tensor_to_display_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
    normalize: str | bool = "imagenet",
) -> np.ndarray:
    """Convert a tensor with shape ``(C, H, W)`` to a displayable RGB image."""
    if not torch.is_tensor(tensor):
        raise TypeError("tensor must be a torch.Tensor.")

    image = tensor.detach().cpu().clone()

    if image.ndim != 3:
        raise ValueError(f"Expected tensor shape (C, H, W), got {tuple(image.shape)}.")

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[0]}.")

    if denormalize:
        if normalize is True:
            normalize = "imagenet"
        if str(normalize).lower().strip() == "imagenet":
            mean, std = get_imagenet_mean_std()
            mean_tensor = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
            std_tensor = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
            image = image * std_tensor + mean_tensor

    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def transform_image_to_array(
    path: str | Path,
    transform: T.Compose,
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    """Apply a transform to one image path and return a NumPy array."""
    if transform is None:
        raise ValueError("transform must not be None.")

    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))

    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy().astype(dtype, copy=False)

    return np.asarray(tensor, dtype=dtype)


__all__ = [
    "SUPPORTED_TRANSFORM_MODES",
    "LetterBox",
    "SquarePad",
    "get_imagenet_mean_std",
    "get_normalize_transform",
    "get_hybrid_transform",
    "get_dl_transform",
    "build_image_transform",
    "tensor_to_display_image",
    "transform_image_to_array",
]
