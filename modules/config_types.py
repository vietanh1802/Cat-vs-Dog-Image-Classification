"""
Typed configuration objects for the cat/dog image-classification project.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


PROJECT_NAME = "ml_image_classifier"
DEFAULT_SEED = 42

SUPPORTED_PREPROCESSING_MODES = ("stretch", "letterbox", "center_crop", "augmented")
SUPPORTED_BACKBONES = ("efficientnet_b0", "resnet18", "vgg16")
SUPPORTED_CLASSIFIERS = (
    "logistic_regression",
    "svm_linear",
    "random_forest",
    "voting_soft",
    "stacking",
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _filter_dataclass_kwargs(cls: type, data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only valid dataclass fields and reject unknown keys."""
    data = data or {}
    field_names = {item.name for item in fields(cls)}

    unknown_keys = set(data.keys()) - field_names
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown_keys)}"
        )

    return {key: value for key, value in data.items() if key in field_names}


def _merge_dict(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Deep merge two dictionaries of the same schema.

    This is safe for params of the SAME classifier, but should not be used to
    merge params across different classifier types.
    """
    result = deepcopy(dict(defaults))

    if not overrides:
        return result

    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


# -----------------------------------------------------------------------------
# Classifier config objects
# -----------------------------------------------------------------------------


@dataclass
class BaseClassifierConfig:
    """Base class for all classifier configs."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    allowed_param_keys: ClassVar[set[str]] = set()
    default_params: ClassVar[dict[str, Any]] = {}

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> "BaseClassifierConfig":
        final_params = _merge_dict(cls.default_params, params)
        config = cls(name=cls.classifier_name(), params=final_params)
        config.validate()
        return config

    @classmethod
    def classifier_name(cls) -> str:
        raise NotImplementedError

    def validate(self) -> None:
        unknown_keys = set(self.params.keys()) - set(self.allowed_param_keys)
        if unknown_keys:
            raise ValueError(
                f"Invalid parameter(s) for classifier '{self.name}': {sorted(unknown_keys)}. "
                f"Allowed keys: {sorted(self.allowed_param_keys)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": deepcopy(self.params),
        }


@dataclass
class LogisticRegressionConfig(BaseClassifierConfig):
    allowed_param_keys: ClassVar[set[str]] = {
        "C",
        "max_iter",
        "n_jobs",
        "solver",
        "penalty",
        "class_weight",
        "random_state",
    }

    default_params: ClassVar[dict[str, Any]] = {
        "C": 1.0,
        "max_iter": 3000,
        "n_jobs": -1,
    }

    @classmethod
    def classifier_name(cls) -> str:
        return "logistic_regression"


@dataclass
class SVMLinearConfig(BaseClassifierConfig):
    allowed_param_keys: ClassVar[set[str]] = {
        "C",
        "probability",
        "class_weight",
        "random_state",
    }

    default_params: ClassVar[dict[str, Any]] = {
        "C": 1.0,
        "probability": True,
    }

    @classmethod
    def classifier_name(cls) -> str:
        return "svm_linear"


@dataclass
class RandomForestConfig(BaseClassifierConfig):
    allowed_param_keys: ClassVar[set[str]] = {
        "n_estimators",
        "max_depth",
        "n_jobs",
        "class_weight",
        "random_state",
        "min_samples_split",
        "min_samples_leaf",
    }

    default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 100,
        "max_depth": None,
        "n_jobs": -1,
    }

    @classmethod
    def classifier_name(cls) -> str:
        return "random_forest"


@dataclass
class VotingSoftConfig(BaseClassifierConfig):
    allowed_param_keys: ClassVar[set[str]] = {
        "voting",
        "n_jobs",
        "lr",
        "svm",
        "rf",
        "weights",
    }

    default_params: ClassVar[dict[str, Any]] = {
        "voting": "soft",
        "n_jobs": -1,
        "lr": {
            "C": 0.1,
            "max_iter": 3000,
            "n_jobs": -1,
        },
        "svm": {
            "C": 0.1,
            "probability": True,
        },
        "rf": {
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": -1,
        },
    }

    @classmethod
    def classifier_name(cls) -> str:
        return "voting_soft"


@dataclass
class StackingConfig(BaseClassifierConfig):
    allowed_param_keys: ClassVar[set[str]] = {
        "n_jobs",
        "lr",
        "svm",
        "rf",
        "final_estimator",
    }

    default_params: ClassVar[dict[str, Any]] = {
        "n_jobs": -1,
        "lr": {
            "C": 0.1,
            "max_iter": 3000,
            "n_jobs": -1,
        },
        "svm": {
            "C": 0.1,
            "probability": True,
        },
        "rf": {
            "n_estimators": 100,
            "max_depth": None,
            "n_jobs": -1,
        },
        "final_estimator": {
            "C": 1.0,
            "max_iter": 3000,
            "n_jobs": -1,
        },
    }

    @classmethod
    def classifier_name(cls) -> str:
        return "stacking"


CLASSIFIER_CONFIG_REGISTRY: dict[str, type[BaseClassifierConfig]] = {
    "logistic_regression": LogisticRegressionConfig,
    "svm_linear": SVMLinearConfig,
    "random_forest": RandomForestConfig,
    "voting_soft": VotingSoftConfig,
    "stacking": StackingConfig,
}


def build_classifier_config(data: Mapping[str, Any] | None = None) -> BaseClassifierConfig:
    """
    Build the correct classifier config object from a dictionary.

    Example:
        {"name": "logistic_regression", "params": {"C": 10.0}}
    """
    data = data or {}

    name = str(data.get("name", "voting_soft")).lower().strip()
    params = data.get("params", None)

    if name not in CLASSIFIER_CONFIG_REGISTRY:
        raise ValueError(
            f"Unsupported classifier.name: {name}. "
            f"Supported classifiers: {list(CLASSIFIER_CONFIG_REGISTRY.keys())}"
        )

    return CLASSIFIER_CONFIG_REGISTRY[name].from_params(params)


# -----------------------------------------------------------------------------
# Non-classifier config objects
# -----------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    name: str = PROJECT_NAME
    task: str = "binary_image_classification"
    target: str = "cat_vs_dog"


@dataclass
class RuntimeConfig:
    deterministic: bool = False
    device: str = "auto"
    num_workers: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "RuntimeConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.num_workers) < 0:
            raise ValueError("runtime.num_workers must be non-negative")


@dataclass
class PathsConfig:
    workspace: str
    data_dir: str
    raw_data_dir: str
    processed_data_dir: str
    features_dir: str
    models_dir: str
    reports_dir: str
    figures_dir: str
    results_dir: str

    @classmethod
    def from_workspace(cls, workspace: str | Path) -> "PathsConfig":
        workspace = Path(workspace)

        return cls(
            workspace=str(workspace),
            data_dir=str(workspace / "data"),
            raw_data_dir=str(workspace / "data" / "raw"),
            processed_data_dir=str(workspace / "data" / "processed"),
            features_dir=str(workspace / "features"),
            models_dir=str(workspace / "models"),
            reports_dir=str(workspace / "reports"),
            figures_dir=str(workspace / "reports" / "figures"),
            results_dir=str(workspace / "reports" / "results"),
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        default_workspace: str | Path = "/content/ml_image_classifier",
    ) -> "PathsConfig":
        data = data or {}

        workspace = data.get("workspace", default_workspace)
        config = cls.from_workspace(workspace)

        for key, value in data.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown path config key: {key}")
            setattr(config, key, str(value))

        return config


@dataclass
class DatasetConfig:
    dataset_id: str = "tongpython/cat-and-dog"
    local_root: str | None = None
    kaggle_input_dir: str = "/kaggle/input"
    extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    class_map: dict[str, int] = field(
        default_factory=lambda: {
            "cat": 0,
            "cats": 0,
            "dog": 1,
            "dogs": 1,
        }
    )
    label_names: dict[str, str] = field(
        default_factory=lambda: {
            "0": "cat",
            "1": "dog",
        }
    )
    drop_unknown: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "DatasetConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))

        # Normalize label_names keys to strings for JSON stability.
        config.label_names = {str(key): value for key, value in config.label_names.items()}

        return config


@dataclass
class SplitConfig:
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    label_col: str = "label"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "SplitConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.validate()
        return config

    def validate(self) -> None:
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]

        if any(float(ratio) < 0 for ratio in ratios):
            raise ValueError(f"Split ratios must be non-negative, got {ratios}")

        total = sum(float(ratio) for ratio in ratios)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got sum={total:.6f}")


@dataclass
class AuditConfig:
    compute_hash: bool = True
    path_col: str = "path"
    label_col: str = "label"
    label_name_col: str = "label_name"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "AuditConfig":
        return cls(**_filter_dataclass_kwargs(cls, data))


@dataclass
class CleaningConfig:
    enabled: bool = False
    remove_corrupted: bool = True
    remove_duplicates: bool = True
    duplicate_hamming_threshold: int = 4
    min_side: int = 64
    max_aspect_extremity: float = 5.0
    min_blur_laplacian: float = 40.0
    min_entropy: float = 0.0
    max_near_mono_ratio: float = 0.92
    max_dark_ratio: float = 0.98
    max_bright_ratio: float = 0.98
    min_mean_sat: float = 0.0
    min_chroma_mean: float = 0.0
    min_center_saliency_ratio: float = 0.0
    max_compression_artifact: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "CleaningConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.validate()
        return config

    def validate(self) -> None:
        if int(self.duplicate_hamming_threshold) < 0:
            raise ValueError("cleaning.duplicate_hamming_threshold must be non-negative")

        numeric_keys = [
            "min_side",
            "max_aspect_extremity",
            "min_blur_laplacian",
            "min_entropy",
            "max_near_mono_ratio",
            "max_dark_ratio",
            "max_bright_ratio",
            "min_mean_sat",
            "min_chroma_mean",
            "min_center_saliency_ratio",
        ]

        for key in numeric_keys:
            value = getattr(self, key)
            if value is not None and float(value) < 0:
                raise ValueError(f"cleaning.{key} must be non-negative")


@dataclass
class PreprocessingConfig:
    mode: str = "augmented"
    image_size: int = 224
    train_augmentation: bool = False
    normalize: str = "imagenet"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "PreprocessingConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.mode = str(config.mode).lower().strip()
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in SUPPORTED_PREPROCESSING_MODES:
            raise ValueError(
                f"Unsupported preprocessing.mode: {self.mode}. "
                f"Supported modes: {list(SUPPORTED_PREPROCESSING_MODES)}"
            )

        if int(self.image_size) <= 0:
            raise ValueError("preprocessing.image_size must be positive")


@dataclass
class FeatureExtractionConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    batch_size: int = 128
    num_workers: int = 2
    data_parallel: bool = False
    force_recompute: bool = False
    file_format: str = "npy"
    on_error: str = "raise"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "FeatureExtractionConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.backbone = str(config.backbone).lower().strip()
        config.file_format = str(config.file_format).lower().strip()
        config.validate()
        return config

    def validate(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported feature_extraction.backbone: {self.backbone}. "
                f"Supported backbones: {list(SUPPORTED_BACKBONES)}"
            )

        if int(self.batch_size) <= 0:
            raise ValueError("feature_extraction.batch_size must be positive")

        if int(self.num_workers) < 0:
            raise ValueError("feature_extraction.num_workers must be non-negative")

        if self.file_format not in {"npy", "h5"}:
            raise ValueError("feature_extraction.file_format must be either 'npy' or 'h5'")


@dataclass
class TuneHyperparameterConfig:
    enabled: bool = True
    grid_size: str = "small"
    cv: int = 3
    scoring: list[str] | str = field(default_factory=lambda: ["f1_macro", "accuracy"])
    refit: str = "f1_macro"
    n_jobs: int = -1
    verbose: int = 2

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "TuneHyperparameterConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.grid_size = str(config.grid_size).lower().strip()
        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            return

        if self.grid_size not in {"small", "large"}:
            raise ValueError("tune_hyperparameter.grid_size must be 'small' or 'large'")

        if int(self.cv) < 2:
            raise ValueError("tune_hyperparameter.cv must be at least 2")

        if int(self.n_jobs) < -1 or int(self.n_jobs) == 0:
            raise ValueError("tune_hyperparameter.n_jobs must be -1 or positive")

        if isinstance(self.scoring, str):
            if not self.scoring.strip():
                raise ValueError("tune_hyperparameter.scoring must not be empty")
            if self.refit != self.scoring:
                raise ValueError("tune_hyperparameter.refit must match scoring when scoring is a string")

        elif isinstance(self.scoring, list):
            if not self.scoring:
                raise ValueError("tune_hyperparameter.scoring must not be empty")
            if self.refit not in self.scoring:
                raise ValueError(
                    f"tune_hyperparameter.refit='{self.refit}' must be one of {self.scoring}"
                )

        else:
            raise ValueError("tune_hyperparameter.scoring must be a string or list of strings")


@dataclass
class DeepLearningConfig:
    enabled: bool = False
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    head_epochs: int = 3
    fine_tune_epochs: int = 2
    head_lr: float = 1e-3
    fine_tune_lr: float = 1e-5
    dropout: float = 0.2
    unfreeze_last_blocks: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "DeepLearningConfig":
        config = cls(**_filter_dataclass_kwargs(cls, data))
        config.backbone = str(config.backbone).lower().strip()
        config.validate()
        return config

    def validate(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported deep_learning.backbone: {self.backbone}. "
                f"Supported backbones: {list(SUPPORTED_BACKBONES)}"
            )

        if int(self.image_size) <= 0:
            raise ValueError("deep_learning.image_size must be positive")

        if int(self.batch_size) <= 0:
            raise ValueError("deep_learning.batch_size must be positive")

        if int(self.head_epochs) < 0:
            raise ValueError("deep_learning.head_epochs must be non-negative")

        if int(self.fine_tune_epochs) < 0:
            raise ValueError("deep_learning.fine_tune_epochs must be non-negative")

        if not 0 <= float(self.dropout) < 1:
            raise ValueError("deep_learning.dropout must be in [0, 1)")

        if float(self.head_lr) <= 0:
            raise ValueError("deep_learning.head_lr must be positive")

        if float(self.fine_tune_lr) <= 0:
            raise ValueError("deep_learning.fine_tune_lr must be positive")


# -----------------------------------------------------------------------------
# Full project config
# -----------------------------------------------------------------------------


@dataclass
class FullConfig:
    project: ProjectConfig
    seed: int
    runtime: RuntimeConfig
    paths: PathsConfig
    dataset: DatasetConfig
    split: SplitConfig
    audit: AuditConfig
    cleaning: CleaningConfig
    preprocessing: PreprocessingConfig
    feature_extraction: FeatureExtractionConfig
    tune_hyperparameter: TuneHyperparameterConfig
    classifier: BaseClassifierConfig
    deep_learning: DeepLearningConfig

    @classmethod
    def from_dict(
        cls,
        user_config: Mapping[str, Any] | None = None,
        default_workspace: str | Path = "/content/ml_image_classifier",
    ) -> "FullConfig":
        user_config = user_config or {}

        return cls(
            project=ProjectConfig(**_filter_dataclass_kwargs(ProjectConfig, user_config.get("project"))),
            seed=int(user_config.get("seed", DEFAULT_SEED)),
            runtime=RuntimeConfig.from_dict(user_config.get("runtime")),
            paths=PathsConfig.from_dict(user_config.get("paths"), default_workspace=default_workspace),
            dataset=DatasetConfig.from_dict(user_config.get("dataset")),
            split=SplitConfig.from_dict(user_config.get("split")),
            audit=AuditConfig.from_dict(user_config.get("audit")),
            cleaning=CleaningConfig.from_dict(user_config.get("cleaning")),
            preprocessing=PreprocessingConfig.from_dict(user_config.get("preprocessing")),
            feature_extraction=FeatureExtractionConfig.from_dict(user_config.get("feature_extraction")),
            tune_hyperparameter=TuneHyperparameterConfig.from_dict(
                user_config.get("tune_hyperparameter", user_config.get("tune-hyperparameter"))
            ),
            classifier=build_classifier_config(user_config.get("classifier")),
            deep_learning=DeepLearningConfig.from_dict(user_config.get("deep_learning")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        # Keep backward compatibility with your current config key.
        data["tune-hyperparameter"] = data.pop("tune_hyperparameter")

        return data
    