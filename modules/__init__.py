"""Internal module package for the image-classification project."""

from __future__ import annotations
from . import artifacts, backbones, classical_models, cleaning, config_types, config_utils, data_utils, datasets, deep_learning, evaluation, feature_extraction, grid_search, image_audit, threshold_experiments, transforms, visualization
  
__version__ = "1.0.0"

__all__ = [
	"__version__",
	"artifacts",
	"backbones",
	"classical_models",
	"cleaning",
  "config_types"
	"config_utils",
	"data_utils",
	"datasets",
  "transforms",
	"deep_learning",
	"evaluation",
	"feature_extraction",
	"grid_search",
	"image_audit",
	"threshold_experiments",
	"visualization",
]
