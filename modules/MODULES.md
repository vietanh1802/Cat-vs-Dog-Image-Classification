# MODULES.md — Module API Reference

This document summarizes the module structure for the cat/dog binary image-classification project.

## Architecture rule

The notebooks should act as the **front-end**: explanation, configuration, execution, visualization, and interpretation.

The `modules/` folder should act as the **backend**: reusable logic with simple public APIs.

General rules:

- `config_utils.py` handles config/runtime only.
- `data_utils.py` handles dataset discovery, dataframe creation, splitting, and summaries.
- `image_audit.py` computes image-quality metrics only.
- `cleaning.py` decides keep/remove based on audit metrics and cleaning config.
- `threshold_experiments.py` runs Sweet Spot threshold experiments.
- `transforms.py` builds image transforms only.
- `datasets.py` defines PyTorch datasets/dataloaders only.
- `backbones.py` loads pretrained backbones and transfer-learning models.
- `feature_extraction.py` extracts frozen deep features only.
- `classical_models.py` builds/trains classical ML classifiers only.
- `deep_learning.py` trains end-to-end transfer-learning models only.
- `evaluation.py` evaluates predictions/models only.
- `grid_search.py` runs config-based classical grid search.
- `visualization.py` plots only.
- `artifacts.py` saves/loads artifacts only.

---

# Folder structure

```text
modules/
├── __init__.py
├── config_utils.py
├── data_utils.py
├── image_audit.py
├── cleaning.py
├── transforms.py
├── backbones.py
├── datasets.py
├── feature_extraction.py
├── classical_models.py
├── deep_learning.py
├── evaluation.py
├── threshold_experiments.py
├── grid_search.py
├── artifacts.py
└── visualization.py
```

---

# `modules/__init__.py`

Package marker for the `modules` folder.

| Item          | Input | Output      | Meaning                             |
| ------------- | ----- | ----------- | ----------------------------------- |
| `__version__` | None  | `str`       | Optional package version string.    |
| `__all__`     | None  | `list[str]` | Optional list of public submodules. |

---

# `modules/config_utils.py`

Configuration and runtime utilities.

| Function / Constant                                            | Input                                | Output                    | Meaning                                                                 |
| -------------------------------------------------------------- | ------------------------------------ | ------------------------- | ----------------------------------------------------------------------- |
| `PROJECT_NAME`                                                 | None                                 | `str`                     | Project folder/name constant.                                           |
| `DEFAULT_SEED`                                                 | None                                 | `int`                     | Default seed used across experiments.                                   |
| `SUPPORTED_PREPROCESSING_MODES`                                | None                                 | `tuple[str, ...]`         | Valid preprocessing modes.                                              |
| `SUPPORTED_BACKBONES`                                          | None                                 | `tuple[str, ...]`         | Valid pretrained backbones.                                             |
| `SUPPORTED_CLASSIFIERS`                                        | None                                 | `tuple[str, ...]`         | Valid classical classifiers.                                            |
| `set_seed(seed=42, deterministic=False)`                       | `seed: int`, `deterministic: bool`   | `None`                    | Sets Python, NumPy, and PyTorch seeds.                                  |
| `get_device()`                                                 | None                                 | `torch.device` or `"cpu"` | Returns CUDA device if available, otherwise CPU.                        |
| `ensure_dirs(*dirs)`                                           | `str/Path/None` values               | `None`                    | Creates directories; ignores `None`.                                    |
| `resolve_workspace(workspace=None, project_name=PROJECT_NAME)` | Optional path/project name           | `Path`                    | Resolves workspace path for local, Colab, or Kaggle.                    |
| `deep_update(default, override=None)`                          | `Mapping`, optional override mapping | `dict`                    | Deep-merges config without mutating original.                           |
| `get_default_config()`                                         | None                                 | `dict[str, Any]`          | Returns fresh default config for the whole project.                     |
| `validate_config(config)`                                      | Config mapping                       | `None` or raises          | Validates config sections and common value ranges.                      |
| `config_to_run_name(config)`                                   | Config mapping                       | `str`                     | Builds filesystem-safe run name from preprocessing/backbone/classifier. |
| `save_config(config, path)`                                    | Config mapping, path                 | `None`                    | Saves config as strict JSON.                                            |
| `load_config(path)`                                            | Path                                 | `dict[str, Any]`          | Loads config from JSON.                                                 |

Private helpers:

| Helper                                                    | Input                      | Output           | Meaning                                                                               |
| --------------------------------------------------------- | -------------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| `_validate_split_config(split)`                           | Split config               | `None` or raises | Validates train/val/test ratios.                                                      |
| `_validate_preprocessing_config(preprocessing)`           | Preprocessing config       | `None` or raises | Validates mode and image size.                                                        |
| `_validate_feature_extraction_config(feature_extraction)` | Feature config             | `None` or raises | Validates backbone, batch size, workers, file format.                                 |
| `_validate_classifier_config(classifier)`                 | Classifier config          | `None` or raises | Validates classifier name.                                                            |
| `_validate_tune_hyperparameter_config(tune)`              | Tune-hyperparameter config | `None` or raises | Validates grid_size, cv, scoring, n_jobs, and enabled flag for hyperparameter tuning. |
| `_validate_runtime_config(runtime)`                       | Runtime config             | `None` or raises | Validates runtime worker settings.                                                    |
| `_validate_cleaning_config(cleaning)`                     | Cleaning config            | `None` or raises | Validates cleaning thresholds.                                                        |
| `_validate_deep_learning_config(deep_learning)`           | DL config                  | `None` or raises | Validates DL backbone, image size, epochs, LR, dropout.                               |
| `_validate_image_size(value, field_name)`                 | Size value and field name  | `None` or raises | Validates integer or 2-tuple image size.                                              |
| `_json_safe(value)`                                       | Any value                  | JSON-safe value  | Converts Path/NumPy/non-finite values for JSON.                                       |

## Hyperparameter Tuning Configuration

The `tune-hyperparameter` section controls grid search behavior independently from classifier selection:

```json
{
  "tune-hyperparameter": {
    "enabled": true, // Enable/disable tuning; if false, validates but skips tuning
    "grid_size": "small", // "small" or "large" - controls parameter grid expansion
    "cv": 3, // Cross-validation folds (≥2)
    "scoring": ["f1_macro", "accuracy"], // Single metric string or ordered list of metrics
    "n_jobs": -1 // Parallel jobs (-1 = all cores, ≥1 = specific count)
  },
  "classifier": {
    "name": "voting_soft",
    "params": {
      // Optional: explicit parameter overrides per classifier
      // If provided, skips default grid generation via get_param_grid()
    }
  }
}
```

**Configuration Flow:**

1. Tuning is enabled by default (`enabled: true`).
2. When enabled, `grid_size` determines parameter search space breadth.
3. `cv` controls cross-validation folds during grid search.
4. `scoring`: Single metric string or ordered list. First metric = refit metric; others = secondary tracked metrics.
5. `n_jobs`: Parallelization (-1 = all cores).

**Classifier Params Hierarchy:**

- If `classifier.params` is provided → use exact values; tuning skips grid search but respects `cv` and `scoring`.
- If `classifier.params` is omitted → generate grid via `get_param_grid(classifier.name, tune-hyperparameter.grid_size)`.

---

# `modules/data_utils.py`

Dataset discovery, dataframe creation, stratified splitting, and dataset summaries.

| Function / Constant                                                                                                              | Input                                                      | Output                                            | Meaning                                                                                    |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `DEFAULT_EXTENSIONS`                                                                                                             | None                                                       | `tuple[str, ...]`                                 | Default image extensions.                                                                  |
| `resolve_dataset_root(dataset_id=None, local_root=None, kaggle_input_dir="/kaggle/input", extensions=None, allow_download=True)` | Dataset ID/local root/Kaggle path/extensions/download flag | `Path`                                            | Finds dataset root from local path, Kaggle input, or KaggleHub.                            |
| `list_image_paths(root, extensions=None)`                                                                                        | Dataset root, extensions                                   | `list[Path]`                                      | Recursively lists supported image files in stable order.                                   |
| `infer_label_from_path(path, class_map=None)`                                                                                    | Image path, optional token-to-label map                    | `tuple[int \| None, str \| None]`                 | Infers numeric label and label name from filename/folder tokens.                           |
| `build_raw_dataframe(root, extensions=None, class_map=None, drop_unknown=True)`                                                  | Dataset root and label options                             | `pd.DataFrame`                                    | Builds dataframe with `sample_id`, `path`, `label`, `label_name`, `filename`, `extension`. |
| `stratified_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, label_col="label")`                               | Dataframe and split settings                               | `tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` | Creates train/val/test dataframes using stratified labels.                                 |
| `sample_by_class(df, n_per_class=5, label_col="label_name", seed=42)`                                                            | Dataframe, sample count, label col, seed                   | `pd.DataFrame`                                    | Returns up to `n_per_class` samples per class.                                             |
| `summarize_class_distribution(df, label_col="label_name")`                                                                       | Dataframe and label column                                 | `pd.DataFrame`                                    | Returns count and percentage by class.                                                     |
| `summarize_split_distribution(train_df, val_df, test_df, label_col="label_name")`                                                | Train/val/test dataframes                                  | `pd.DataFrame`                                    | Convenience wrapper for train/val/test class distribution.                                 |
| `summarize_splits_distribution(splits, label_col="label_name")`                                                                  | Mapping split name → dataframe                             | `pd.DataFrame`                                    | Generic split distribution summary.                                                        |
| `read_image_metadata(path)`                                                                                                      | Image path                                                 | `dict[str, Any]`                                  | Reads width, height, channels, aspect ratio, min/max side.                                 |
| `add_image_metadata(df, path_col="path")`                                                                                        | Dataframe with image paths                                 | `pd.DataFrame`                                    | Adds image metadata columns to dataframe.                                                  |
| `remove_spatial_outliers(df, cols=("width", "height"), n_std=3.0)`                                                               | Dataframe with width/height                                | `pd.DataFrame`                                    | Removes spatial outliers using mean ± n_std rule. Borderline cleaning helper.              |

Private helpers:

| Helper                                                  | Input                   | Output            | Meaning                                                    |
| ------------------------------------------------------- | ----------------------- | ----------------- | ---------------------------------------------------------- |
| `_normalize_extensions(extensions)`                     | Extension iterable      | `tuple[str, ...]` | Normalizes extensions to dotted lowercase suffixes.        |
| `_contains_images(root, extensions=None)`               | Directory path          | `bool`            | Checks whether root contains at least one supported image. |
| `_download_kaggle_dataset(dataset_id)`                  | Kaggle dataset ID       | `Path`            | Downloads dataset using KaggleHub.                         |
| `_default_class_map()`                                  | None                    | `dict[str, int]`  | Default cat/dog token map.                                 |
| `_canonical_label_names(class_map)`                     | Token-to-label map      | `dict[int, str]`  | Infers readable label names.                               |
| `_tokenize_path_part(text)`                             | Path string component   | `list[str]`       | Tokenizes a path component.                                |
| `_candidate_label_parts(path)`                          | Path                    | `list[str]`       | Returns filename/parents in label-search order.            |
| `_deduplicate_preserve_order(paths)`                    | Path sequence           | `list[Path]`      | Removes duplicate paths while preserving order.            |
| `_coerce_label_name(label, label_names)`                | Label and label map     | `str \| None`     | Converts numeric label to readable name.                   |
| `_validate_min_class_count(df, label_col, min_count=2)` | Dataframe and label col | `None` or raises  | Recommended helper for friendlier stratified-split errors. |

---

# `modules/artifacts.py`

Artifact saving/loading utilities.

| Function                                                                 | Input                                   | Output                          | Meaning                                                   |
| ------------------------------------------------------------------------ | --------------------------------------- | ------------------------------- | --------------------------------------------------------- |
| `save_json(data, path, indent=2)`                                        | JSON-like data, path                    | `None`                          | Saves JSON after converting ML values to JSON-safe types. |
| `load_json(path)`                                                        | Path                                    | `dict`                          | Loads JSON file.                                          |
| `save_dataframe(df, path, index=False)`                                  | DataFrame, path                         | `None`                          | Saves dataframe to `.csv` or `.parquet`.                  |
| `load_dataframe(path)`                                                   | Path                                    | `pd.DataFrame`                  | Loads dataframe from `.csv` or `.parquet`.                |
| `save_numpy(array, path)`                                                | NumPy array, path                       | `None`                          | Saves `.npy` array.                                       |
| `load_numpy(path)`                                                       | Path                                    | `np.ndarray`                    | Loads `.npy` with `allow_pickle=False`.                   |
| `save_pickle(obj, path)`                                                 | Trusted Python object, path             | `None`                          | Saves trusted object to pickle.                           |
| `save_feature_split(X, y, split_name, output_dir)`                       | Feature matrix, labels, split name, dir | `dict[str, Path]`               | Saves `X_<split>.npy` and `y_<split>.npy`.                |
| `load_feature_split(split_name, feature_dir)`                            | Split name and feature dir              | `tuple[np.ndarray, np.ndarray]` | Loads cached feature split.                               |
| `feature_files_exist(feature_dir, split_names=("train", "val", "test"))` | Feature dir and split names             | `bool`                          | Checks whether all cached feature files exist.            |

Private helpers:

| Helper                 | Input      | Output          | Meaning                                                         |
| ---------------------- | ---------- | --------------- | --------------------------------------------------------------- |
| `_ensure_parent(path)` | File path  | `None`          | Creates parent directory.                                       |
| `_to_jsonable(obj)`    | Any object | JSON-safe value | Converts Path, NumPy scalar/array, dict/list/tuple recursively. |

---

# `modules/backbones.py`

Pretrained backbone and transfer-model construction.

| Function / Constant                                                                                                 | Input                             | Output            | Meaning                                                                             |
| ------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------- | ----------------------------------------------------------------------------------- |
| `_SUPPORTED_BACKBONES`                                                                                              | None                              | `tuple[str, ...]` | Supported backbone names.                                                           |
| `list_supported_backbones()`                                                                                        | None                              | `list[str]`       | Returns supported backbones.                                                        |
| `get_backbone(name, device=None, pretrained=True, freeze=True, data_parallel=False)`                                | Backbone name and runtime options | `nn.Module`       | Loads pretrained backbone with classification head stripped for feature extraction. |
| `get_feature_dim(backbone_name)`                                                                                    | Backbone name                     | `int`             | Returns output embedding dimension.                                                 |
| `freeze_model(model)`                                                                                               | PyTorch model                     | `nn.Module`       | Sets all parameters `requires_grad=False`.                                          |
| `unfreeze_model(model)`                                                                                             | PyTorch model                     | `nn.Module`       | Sets all parameters `requires_grad=True`.                                           |
| `replace_classifier_head(model, backbone_name, num_classes, dropout=0.2)`                                           | Full torchvision model            | `nn.Module`       | Replaces model classifier head.                                                     |
| `build_transfer_model(backbone_name, num_classes, pretrained=True, dropout=0.2, freeze_backbone=True, device=None)` | Backbone config and class count   | `nn.Module`       | Builds transfer-learning model for end-to-end DL.                                   |

Private helpers:

| Helper                                    | Input                             | Output      | Meaning                                         |
| ----------------------------------------- | --------------------------------- | ----------- | ----------------------------------------------- |
| `_load_base_model(name, pretrained=True)` | Backbone name and pretrained flag | `nn.Module` | Loads full torchvision base model.              |
| `_strip_classifier(model, name)`          | Full model and backbone name      | `nn.Module` | Removes classifier head for feature extraction. |

---

# `modules/datasets.py`

PyTorch Dataset and DataLoader utilities.

| Function / Class                                                                                                                                                              | Input                                        | Output                  | Meaning                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------- | -------------------------------------------------------------- |
| `ImagePathDataset(df=None, paths=None, labels=None, transform=None, path_col="path", label_col="label", return_labels=True, fallback_size=224, on_error="raise")`             | DataFrame or paths/labels plus transform     | `Dataset`               | Reads images from paths and returns image or `(image, label)`. |
| `NpyBatchDataset(batch_dir, split_name, mmap_mode="r", return_tensors=True)`                                                                                                  | Feature cache dir and split name             | `Dataset`               | Loads saved `X_<split>.npy` / `y_<split>.npy`.                 |
| `create_image_dataloader(df, transform, batch_size, shuffle=False, num_workers=0, pin_memory=True, path_col="path", label_col="label", return_labels=True, on_error="raise")` | Dataframe and dataloader settings            | `DataLoader`            | Creates one image dataloader.                                  |
| `create_image_dataloaders(splits, transforms, batch_size, num_workers=0, pin_memory=True, path_col="path", label_col="label", return_labels=True, on_error="raise")`          | Mapping split → df and transform map         | `dict[str, DataLoader]` | Creates dataloaders for train/val/test or any split mapping.   |
| `create_npy_batch_dataloader(batch_dir, split_name, batch_size, shuffle=False, num_workers=0, pin_memory=True, mmap_mode="r", return_tensors=True)`                           | Feature dir, split name, dataloader settings | `DataLoader`            | Creates dataloader for cached NumPy features.                  |

Private helpers:

| Helper                                             | Input                            | Output           | Meaning                                      |
| -------------------------------------------------- | -------------------------------- | ---------------- | -------------------------------------------- |
| `_validate_dataframe_columns(df, required_cols)`   | Dataframe and required cols      | `None` or raises | Validates dataframe columns.                 |
| `_read_pil_image(path)`                            | Path                             | `Image.Image`    | Reads image as RGB PIL image.                |
| `_validate_batch_size(batch_size)`                 | Batch size                       | `None` or raises | Validates positive batch size.               |
| `_resolve_split_transform(split_name, transforms)` | Split name and transform mapping | Transform        | Resolves transform with train/eval fallback. |

---

# `modules/transforms.py`

Image preprocessing transforms.

| Function / Class / Constant                                                     | Input                                 | Output                            | Meaning                                                        |
| ------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------- | -------------------------------------------------------------- |
| `SUPPORTED_TRANSFORM_MODES`                                                     | None                                  | `tuple[str, ...]`                 | Valid transform modes.                                         |
| `LetterBox(size, fill=(0,0,0), interpolation=Image.Resampling.LANCZOS)`         | Target size and fill                  | Callable transform                | Resize while preserving aspect ratio, then pad to target size. |
| `SquarePad(fill=(0,0,0))`                                                       | Fill color                            | Callable transform                | Pads image to square without resizing.                         |
| `get_imagenet_mean_std()`                                                       | None                                  | `tuple[list[float], list[float]]` | Returns ImageNet mean/std.                                     |
| `get_normalize_transform(normalize="imagenet")`                                 | `"imagenet"`, `"none"`, bool          | Transform                         | Returns ImageNet Normalize or identity transform.              |
| `get_hybrid_transform(mode, image_size=224, train=False, normalize="imagenet")` | Mode, size, train flag, normalization | `T.Compose`                       | Builds preprocessing transform for frozen feature extraction.  |
| `get_dl_transform(image_size=224, train=False, normalize="imagenet")`           | Size, train flag, normalization       | `T.Compose`                       | Builds end-to-end DL transform using augmentation for train.   |
| `build_image_transform(config, split="train")`                                  | Preprocessing config, split name      | `T.Compose`                       | Config-driven image transform builder.                         |
| `tensor_to_display_image(tensor, denormalize=True, normalize="imagenet")`       | Tensor `(C,H,W)`                      | `np.ndarray`                      | Converts tensor to displayable RGB array.                      |
| `transform_image_to_array(path, transform, dtype="float32")`                    | Image path and transform              | `np.ndarray`                      | Applies transform to one image and returns NumPy array.        |

Private helpers:

| Helper                             | Input               | Output              | Meaning                                  |
| ---------------------------------- | ------------------- | ------------------- | ---------------------------------------- |
| `_validate_image_size(image_size)` | Int or 2-tuple/list | `tuple[int, int]`   | Validates and returns `(height, width)`. |
| `_normalize_mode(mode)`            | Mode string         | `str`               | Normalizes and validates transform mode. |
| `_validate_fill(fill)`             | Fill value          | PIL-compatible fill | Validates padding fill color.            |

---

# `modules/image_audit.py`

Image-quality metric computation.

| Function / Constant                                                                                                                               | Input                                | Output                    | Meaning                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------- | -------------------------------------------------- |
| `DEFAULT_AUDIT_METRICS`                                                                                                                           | None                                 | `tuple[str, ...]`         | Default metric list for summaries.                 |
| `read_image_cv2(path)`                                                                                                                            | Path                                 | `np.ndarray \| None`      | Reads image as BGR OpenCV array.                   |
| `read_image_pil(path)`                                                                                                                            | Path                                 | `Image.Image \| None`     | Reads image as RGB PIL image.                      |
| `compute_laplacian_variance(gray)`                                                                                                                | Grayscale image                      | `float`                   | Blur proxy; lower means blurrier.                  |
| `compute_entropy(gray)`                                                                                                                           | Grayscale image                      | `float`                   | Shannon entropy of grayscale histogram.            |
| `compute_brightness_stats(gray)`                                                                                                                  | Grayscale image                      | `dict[str, float]`        | Brightness mean/std.                               |
| `compute_gray_tolerance(rgb)`                                                                                                                     | RGB image                            | `dict[str, float]`        | Channel-difference metrics for grayscale-likeness. |
| `compute_near_mono_metrics(gray)`                                                                                                                 | Grayscale image                      | `dict[str, float]`        | Dark/bright/near-monochrome ratios.                |
| `compute_aspect_metrics(width, height)`                                                                                                           | Width and height                     | `dict[str, float \| int]` | Size, min/max side, aspect ratio/extremity.        |
| `compute_saturation_metrics(bgr)`                                                                                                                 | BGR image                            | `dict[str, float]`        | HSV saturation mean/std.                           |
| `compute_chromaticity_metrics(rgb)`                                                                                                               | RGB image                            | `dict[str, float]`        | Color variation / chromaticity spread.             |
| `compute_center_saliency(gray)`                                                                                                                   | Grayscale image                      | `dict[str, float]`        | Sobel-based center-saliency proxy.                 |
| `compute_compression_artifact(gray)`                                                                                                              | Grayscale image                      | `float`                   | Approximate block-compression artifact score.      |
| `compute_phash(path)`                                                                                                                             | Image path                           | `str \| None`             | Perceptual hash using `imagehash` if installed.    |
| `inspect_image(path, label=None, label_name=None, compute_hash=True)`                                                                             | Image path and optional label info   | `dict[str, Any]`          | Computes complete audit record for one image.      |
| `audit_dataframe(df, path_col="path", label_col="label", label_name_col="label_name", compute_hash=True, preserve_cols=None, show_progress=True)` | Image dataframe                      | `pd.DataFrame`            | Runs audit over dataframe rows.                    |
| `describe_audit_metrics(audit_df, metrics=None)`                                                                                                  | Audit dataframe and optional metrics | `pd.DataFrame`            | Returns descriptive stats for audit metrics.       |

Private helpers:

| Helper                                 | Input               | Output           | Meaning                                           |
| -------------------------------------- | ------------------- | ---------------- | ------------------------------------------------- |
| `_none_if_missing(value)`              | Any value           | Value or `None`  | Converts pandas missing values to `None`.         |
| `_nan_record(path, label, label_name)` | Path and label info | `dict[str, Any]` | Full audit record for corrupted/unreadable image. |

---

# `modules/cleaning.py`

Image-cleaning decisions from audit metrics.

| Function / Class                                                                 | Input                        | Output                              | Meaning                                                       |
| -------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| `_UnionFind`                                                                     | Internal class               | Internal                            | Duplicate clustering structure.                               |
| `compute_quality_score(audit_df)`                                                | Audit dataframe              | `pd.DataFrame`                      | Adds `quality_score` used to choose duplicate keepers.        |
| `mark_near_duplicates(audit_df, hamming_threshold=4, hash_col="phash", bands=8)` | Audit dataframe              | `pd.DataFrame`                      | Adds duplicate cluster flags and representative/remove flags. |
| `compute_soft_flags(audit_df, cleaning_config)`                                  | Audit df and cleaning config | `pd.DataFrame`                      | Adds threshold-based `flag_*` columns without dropping rows.  |
| `build_cleaning_mask(audit_df, cleaning_config)`                                 | Audit df and cleaning config | `pd.Series[bool]`                   | Returns keep-mask; `True` means keep.                         |
| `assign_removal_reasons(audit_df, cleaning_config)`                              | Audit df and cleaning config | `pd.DataFrame`                      | Adds `keep`, `removal_reason`, `removal_reasons`.             |
| `apply_cleaning(audit_df, cleaning_config)`                                      | Audit df and config          | `tuple[pd.DataFrame, pd.DataFrame]` | Returns `(clean_df, removed_df)`.                             |
| `summarize_cleaning(raw_df, clean_df, removed_df, label_col="label_name")`       | Raw/clean/removed dataframes | `pd.DataFrame`                      | Summarizes before/after/removed counts by class.              |
| `summarize_removal_reasons(removed_df, reason_col="removal_reasons")`            | Removed dataframe            | `pd.DataFrame`                      | Counts removal reasons and percentages.                       |
| `evaluate_cleaning_retention(raw_df, clean_df, label_col="label")`               | Raw and clean dataframes     | `dict[str, Any]`                    | Computes retention/removal/class-balance shift.               |

Private helpers:

| Helper                                                 | Input                        | Output         | Meaning                                |
| ------------------------------------------------------ | ---------------------------- | -------------- | -------------------------------------- |
| `_is_finite_number(value)`                             | Any value                    | `bool`         | Checks finite numeric value.           |
| `_config_value(config, key, default)`                  | Config, key, default         | `float`        | Reads numeric threshold safely.        |
| `_boolean_series(index, value=False)`                  | Dataframe index              | `pd.Series`    | Creates aligned bool series.           |
| `_numeric_series(df, column, default=np.nan)`          | Dataframe and column         | `pd.Series`    | Reads numeric column safely.           |
| `_rank_score(df, col, ascending=True)`                 | Dataframe, column, direction | `pd.Series`    | Percentile quality score component.    |
| `_valid_hex_hash(value)`                               | Hash value                   | `str \| None`  | Validates/normalizes phash.            |
| `_hamming_distance_hex(hash_a, hash_b)`                | Two hashes                   | `int`          | Hamming distance between hex hashes.   |
| `_ensure_duplicate_columns(audit_df, cleaning_config)` | Audit df and config          | `pd.DataFrame` | Adds duplicate columns when requested. |

---

# `modules/feature_extraction.py`

Frozen-backbone feature extraction.

| Function                                                                                                                                                                                                                                                                                                | Input                                           | Output                          | Meaning                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `extract_features(df, transform, backbone_name, batch_size=128, device=None, num_workers=0, path_col="path", label_col="label", pretrained=True, data_parallel=False, show_progress=True, on_error="raise")`                                                                                            | Image dataframe, transform, backbone config     | `tuple[np.ndarray, np.ndarray]` | Extracts feature matrix `X` and label vector `y`.      |
| `extract_feature_splits(train_df, val_df, test_df, train_transform, eval_transform, backbone_name, batch_size=128, device=None, num_workers=0, output_dir=None, pretrained=True, data_parallel=False, force_recompute=False, path_col="path", label_col="label", show_progress=True, on_error="raise")` | Train/val/test dataframes and extraction config | `dict[str, Any]`                | Extracts or loads cached features for all splits.      |
| `load_or_extract_feature_splits(splits, transforms, config, output_dir=None)`                                                                                                                                                                                                                           | Split mapping, transforms, config               | `dict[str, Any]`                | Config-driven wrapper around `extract_feature_splits`. |

Private helpers:

| Helper                                                                                | Input                       | Output           | Meaning                                               |
| ------------------------------------------------------------------------------------- | --------------------------- | ---------------- | ----------------------------------------------------- |
| `_as_device(device=None)`                                                             | Device string/object/None   | `torch.device`   | Normalizes device.                                    |
| `_validate_positive_int(value, name, allow_zero=False)`                               | Value and name              | `int`            | Validates integer hyperparameter.                     |
| `_validate_feature_dataframe(df, path_col, label_col)`                                | Dataframe and column names  | `None` or raises | Validates dataframe columns.                          |
| `_feature_cache_manifest_path(output_dir)`                                            | Output dir                  | `Path`           | Returns manifest path.                                |
| `_build_feature_manifest(backbone_name, pretrained, split_sizes)`                     | Extraction metadata         | `dict`           | Builds cache metadata.                                |
| `_write_feature_manifest(output_dir, manifest)`                                       | Output dir and metadata     | `None`           | Writes feature cache manifest.                        |
| `_read_feature_manifest(output_dir)`                                                  | Output dir                  | `dict \| None`   | Reads feature cache manifest.                         |
| `_cache_matches_request(output_dir, backbone_name, pretrained, expected_split_sizes)` | Cache info and request info | `bool`           | Checks whether feature cache matches requested setup. |
| `_load_cached_feature_splits(output_dir)`                                             | Feature dir                 | `dict[str, Any]` | Loads train/val/test feature splits.                  |

---

# `modules/classical_models.py`

Classical ML classifier construction, training, benchmarking, and tuning.

| Function / Constant                                                                                                                                                         | Input                                     | Output                                          | Meaning                                                                              |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| `_SUPPORTED_CLASSIFIERS`                                                                                                                                                    | None                                      | `tuple[str, ...]`                               | Supported classifier names.                                                          |
| `list_supported_classifiers()`                                                                                                                                              | None                                      | `list[str]`                                     | Returns supported classifiers.                                                       |
| `get_classifier(name, seed=42, **params)`                                                                                                                                   | Classifier name and parameters            | `BaseEstimator`                                 | Builds configured sklearn classifier.                                                |
| `get_param_grid(classifer_name, grid_size="small")`                                                                                                                         | Classifier name and grid size             | `dict[str, Any]`                                | Returns a default hyperparameter grid for the requested classifier.                  |
| `train_classifier(X_train, y_train, classifier_name, seed=42, **params)`                                                                                                    | Feature matrix, labels, classifier config | `BaseEstimator`                                 | Fits a classical classifier.                                                         |
| `benchmark_classifiers(X_train, y_train, X_val, y_val, classifier_configs, seed=42)`                                                                                        | Train/val features and model configs      | `tuple[pd.DataFrame, dict[str, BaseEstimator]]` | Trains multiple classifiers and returns metrics/model dict.                          |
| `select_best_model(results_df, trained_models, metric="f1_macro", model_key_col="run_name")`                                                                                | Results table and model dict              | `tuple[str, BaseEstimator, pd.Series]`          | Selects best trained model.                                                          |
| `tune_with_params(X_train, y_train, classifier_name, param_grid=None, grid_size="small", cv=3, seed=42, scoring="f1_macro", n_jobs=-1, verbose=1, return_train_score=True)` | Training data and tuning parameters       | `GridSearchCV`                                  | Runs grid search with optional default-grid lookup and ordered multi-metric scoring. |
| `tune_classifier_grid(X_train, y_train, classifier_name, param_grid, cv=3, seed=42, scoring="f1_macro", n_jobs=-1)`                                                         | Training data and hyperparameter grid     | `GridSearchCV`                                  | Runs sklearn grid search for one classifier.                                         |

Private helpers:

| Helper                                                             | Input                       | Output                   | Meaning                                        |
| ------------------------------------------------------------------ | --------------------------- | ------------------------ | ---------------------------------------------- |
| `_normalize_classifier_name(name)`                                 | Name string                 | `str`                    | Normalizes/validates classifier name.          |
| `_build_logistic_regression(seed, **params)`                       | Seed and params             | `Pipeline`               | Builds LR with scaler.                         |
| `_build_svm_linear(seed, **params)`                                | Seed and params             | `Pipeline`               | Builds linear SVM with scaler.                 |
| `_build_random_forest(seed, **params)`                             | Seed and params             | `RandomForestClassifier` | Builds RF classifier.                          |
| `_classifier_run_name(cfg, index)`                                 | Classifier config and index | `str`                    | Builds unique run name for benchmark configs.  |
| `_normalize_param_grid_for_estimator(classifier_name, param_grid)` | Classifier name and grid    | `dict`                   | Adds `clf__` prefixes for Pipeline estimators. |

---

# `modules/deep_learning.py`

End-to-end transfer-learning utilities.

| Function                                                                 | Input                                     | Output                                      | Meaning                                                                        |
| ------------------------------------------------------------------------ | ----------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------ |
| `unfreeze_last_blocks(model, num_blocks=1)`                              | Model and number of blocks                | `nn.Module`                                 | Unfreezes last parameterized child modules.                                    |
| `create_image_dataloaders_for_config(train_df, val_df, test_df, config)` | Data splits and DL config                 | `dict[str, DataLoader]`                     | Creates train/val/test DL dataloaders.                                         |
| `build_model_for_config(config, num_classes, device=None)`               | DL config, class count, device            | `nn.Module`                                 | Builds transfer-learning model from config.                                    |
| `build_optimizer(model, config)`                                         | Model and optimizer config                | `torch.optim.Optimizer`                     | Builds optimizer over trainable params only.                                   |
| `build_scheduler(optimizer, config)`                                     | Optimizer and scheduler config            | Scheduler or `None`                         | Builds optional LR scheduler.                                                  |
| `train_one_epoch(model, dataloader, criterion, optimizer, device)`       | Model, data, criterion, optimizer, device | `dict[str, float]`                          | Runs one training epoch.                                                       |
| `evaluate_one_epoch(model, dataloader, criterion, device)`               | Model, data, criterion, device            | `dict[str, float]`                          | Runs one evaluation epoch.                                                     |
| `fit_transfer_model(model, dataloaders, config)`                         | Model, dataloader mapping, config         | `dict[str, Any]`                            | Trains head and optional fine-tuning phase; returns model/history/best metric. |
| `predict_dataloader(model, dataloader, device)`                          | Model, dataloader, device                 | `tuple[np.ndarray, np.ndarray, np.ndarray]` | Returns `y_true`, `y_pred`, `y_prob`.                                          |
| `save_checkpoint(model, optimizer, epoch, metrics, path)`                | Training state and path                   | `None`                                      | Saves PyTorch checkpoint.                                                      |
| `load_checkpoint(path, model, optimizer=None, map_location=None)`        | Checkpoint path and model                 | `dict[str, Any]`                            | Loads checkpoint into model/optimizer.                                         |
| `count_trainable_parameters(model)`                                      | Model                                     | `dict[str, int \| float]`                   | Counts total/trainable/frozen params and trainable percentage.                 |

Private helpers:

| Helper                                                  | Input                            | Output               | Meaning                                         |
| ------------------------------------------------------- | -------------------------------- | -------------------- | ----------------------------------------------- |
| `_as_device(device=None)`                               | Device string/object/None        | `torch.device`       | Normalizes device.                              |
| `_validate_positive_int(value, name, allow_zero=False)` | Value and field name             | `int`                | Validates integer hyperparameter.               |
| `_trainable_parameters(model)`                          | Model                            | `list[nn.Parameter]` | Returns trainable parameters or raises if none. |
| `_clone_state_dict_to_cpu(model)`                       | Model                            | `dict[str, Tensor]`  | Clones best state to CPU.                       |
| `_unwrap_model(model)`                                  | Model                            | `nn.Module`          | Unwraps DataParallel/DDP model.                 |
| `_parameterized_children(model)`                        | Model                            | `list[nn.Module]`    | Direct child modules with parameters.           |
| `_compute_metrics(loss_sum, n_samples, y_true, y_pred)` | Accumulated values               | `dict[str, float]`   | Computes epoch loss/accuracy/F1.                |
| `_scheduler_step(scheduler, val_metrics)`               | Scheduler and validation metrics | `None`               | Steps scheduler, supports ReduceLROnPlateau.    |

---

# `modules/evaluation.py`

Classification evaluation utilities.

| Function                                                                                   | Input                                                 | Output                       | Meaning                                                                 |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| `compute_classification_metrics(y_true, y_pred, y_prob=None)`                              | True labels, predicted labels, optional probabilities | `dict[str, float \| int]`    | Computes accuracy, precision, recall, F1, error rate, optional ROC-AUC. |
| `classification_report_df(y_true, y_pred, labels=None, label_names=None)`                  | True/pred labels and optional labels/names            | `pd.DataFrame`               | Returns sklearn classification report as dataframe.                     |
| `confusion_matrix_df(y_true, y_pred, labels=None, label_names=None)`                       | True/pred labels and optional labels/names            | `pd.DataFrame`               | Returns confusion matrix dataframe.                                     |
| `evaluate_predictions(y_true, y_pred, y_prob=None, labels=None, label_names=None)`         | Prediction arrays and label info                      | `dict[str, Any]`             | Packages metrics, report, and confusion matrix.                         |
| `evaluate_estimator(model, X, y, model_name="model")`                                      | Fitted estimator, features, labels                    | `dict[str, Any]`             | Predicts and evaluates fitted estimator.                                |
| `evaluate_model`                                                                           | Alias                                                 | Same as `evaluate_estimator` | Backward-compatible alias.                                              |
| `find_wrong_predictions(df, y_true, y_pred, y_prob=None, path_col="path", label_map=None)` | Test dataframe and predictions                        | `pd.DataFrame`               | Returns misclassified rows with labels/confidence.                      |
| `format_metrics_table(metrics_dict)`                                                       | Metrics mapping                                       | `pd.DataFrame`               | Converts metrics dict to one-row dataframe.                             |
| `compare_pipeline_results(results)`                                                        | Mapping pipeline name → result/metrics                | `pd.DataFrame`               | Compares multiple pipeline metrics in one table.                        |

Private helpers:

| Helper                                                                     | Input             | Output                   | Meaning                                            |
| -------------------------------------------------------------------------- | ----------------- | ------------------------ | -------------------------------------------------- |
| `_as_1d_array(values, name)`                                               | Array-like values | `np.ndarray`             | Converts to non-empty 1D array.                    |
| `_validate_prediction_lengths(y_true, y_pred)`                             | Arrays            | `None` or raises         | Checks same length.                                |
| `_resolve_labels_and_names(y_true, y_pred, labels=None, label_names=None)` | Labels and names  | `tuple[list, list[str]]` | Resolves stable labels/display names.              |
| `_extract_positive_class_scores(y_prob)`                                   | Probability array | `np.ndarray \| None`     | Extracts positive-class scores for binary ROC-AUC. |

---

# `modules/threshold_experiments.py`

Sweet Spot Threshold experiment utilities.

| Function                                                                                                                                                                                                                      | Input                                                      | Output                              | Meaning                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------- |
| `get_default_threshold_specs()`                                                                                                                                                                                               | None                                                       | `list[dict[str, Any]]`              | Returns single-metric threshold sweep specs.                        |
| `get_default_cleaning_presets()`                                                                                                                                                                                              | None                                                       | `dict[str, dict[str, Any]]`         | Returns cleaning presets using canonical `cleaning.py` config keys. |
| `build_single_metric_mask(audit_df, metric, threshold, direction, base_mask=None)`                                                                                                                                            | Audit df, metric, threshold, direction, optional base mask | `pd.Series[bool]`                   | Builds one keep-mask from a single metric threshold.                |
| `make_proxy_split(audit_df, valid_mask, val_size, seed, label_col="label")`                                                                                                                                                   | Audit df, valid mask, split config                         | `tuple[np.ndarray, np.ndarray]`     | Creates row-position train/val indices for proxy protocol.          |
| `evaluate_cleaning_mask(mask, X_all, y_all, audit_df, train_indices, val_indices, mask_name="mask", threshold="threshold", seed=42, classifier_C=0.1, retention_penalty=0.10, imbalance_penalty=0.05, min_train_samples=100)` | Mask, features/labels, audit df, indices, scoring config   | `dict[str, Any]`                    | Evaluates one cleaning mask with fixed proxy validation.            |
| `run_single_metric_threshold_sweep(X, y, audit_df, threshold_specs, train_indices, val_indices, base_mask, proxy_config=None, score_config=None, seed=42)`                                                                    | Features/labels/audit/sweep specs                          | `pd.DataFrame`                      | Evaluates many single-metric thresholds.                            |
| `evaluate_cleaning_presets(X, y, audit_df, presets, train_indices, val_indices, proxy_config=None, score_config=None, seed=42)`                                                                                               | Features/labels/audit/presets                              | `pd.DataFrame`                      | Evaluates named cleaning presets.                                   |
| `run_cleaning_stability_check(X, y, audit_df, presets, seeds, val_size, proxy_config=None, score_config=None)`                                                                                                                | Features/labels/audit/presets/seeds                        | `tuple[pd.DataFrame, pd.DataFrame]` | Runs stability check across random proxy splits.                    |
| `select_cleaning_policy(stability_summary, presets, min_delta_f1, min_retention_pct, max_imbalance_shift, default_policy)`                                                                                                    | Stability summary and criteria                             | `tuple[str, dict, pd.Series, str]`  | Selects final cleaning preset.                                      |
| `build_cleaning_report_payload(selected_preset, selected_config, selected_row, selection_rule, audit_df, final_clean_df, removed_df, config)`                                                                                 | Selection and output info                                  | `dict[str, Any]`                    | Builds JSON-friendly cleaning report payload.                       |

Private helpers:

| Helper                                     | Input                             | Output                          | Meaning                                         |
| ------------------------------------------ | --------------------------------- | ------------------------------- | ----------------------------------------------- |
| `_normalize_direction(direction)`          | Direction string                  | `str`                           | Normalizes aliases like `min/max` to `gte/lte`. |
| `_validate_arrays(X_all, y_all, audit_df)` | Feature/label arrays and audit df | `tuple[np.ndarray, np.ndarray]` | Validates row alignment.                        |
| `_mask_to_array(mask, audit_df)`           | Mask and audit df                 | `np.ndarray[bool]`              | Converts mask to row-order boolean array.       |
| `_class_balance_shift(y_before, y_after)`  | Label arrays                      | `float`                         | Max absolute change in class proportions.       |
| `_proxy_classifier(classifier_C, seed)`    | C and seed                        | `Pipeline`                      | Builds fixed proxy LR classifier.               |
| `_invalid_mask_result(...)`                | Mask metadata                     | `dict[str, Any]`                | Standard invalid result row.                    |

---

# `modules/grid_search.py`

Classical pipeline grid-search utilities.

| Function                                                                                                         | Input                                         | Output                          | Meaning                                                             |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------- | ------------------------------------------------------------------- |
| `generate_experiment_grid(search_space)`                                                                         | Mapping config key → candidate values         | `list[dict[str, Any]]`          | Generates grid of experiment override dicts.                        |
| `merge_experiment_config(base_config, experiment_config)`                                                        | Base config and overrides                     | `dict[str, Any]`                | Applies overrides, including dotted keys like `classifier.name`.    |
| `run_single_classical_experiment(train_df, val_df, test_df, config, feature_dir, device=None, run_name=None)`    | Split dfs, config, feature dir                | `dict[str, Any]`                | Runs one classical config: transform → features → train → validate. |
| `run_grid_search(train_df, val_df, test_df, base_config, search_space, output_dir, device=None, fail_fast=True)` | Splits, base config, search space, output dir | `pd.DataFrame`                  | Runs all experiments and saves `grid_results.csv`.                  |
| `rank_grid_results(results_df, primary_metric="f1_macro", tie_breakers=None)`                                    | Grid results and ranking settings             | `pd.DataFrame`                  | Ranks successful grid results.                                      |
| `select_default_config(ranked_results, base_config, config_columns=None)`                                        | Ranked results and base config                | `dict[str, Any]`                | Builds final default config from best grid row.                     |
| `select_default_config_from_grid`                                                                                | Alias                                         | Same as `select_default_config` | Backward-compatible clearer alias.                                  |

Private helpers:

| Helper                                               | Input                       | Output                        | Meaning                                               |
| ---------------------------------------------------- | --------------------------- | ----------------------------- | ----------------------------------------------------- |
| `_as_list(value)`                                    | Any value                   | `list[Any]`                   | Treats scalar/string as one candidate.                |
| `_set_by_dotted_key(config, dotted_key, value)`      | Config, dotted key, value   | `None`                        | Sets nested config value.                             |
| `_safe_slug(value)`                                  | Any value                   | `str`                         | Filesystem-safe string.                               |
| `_experiment_run_name(experiment_config, index)`     | Experiment config and index | `str`                         | Stable run name.                                      |
| `_feature_cache_key(config)`                         | Config                      | `str`                         | Feature-cache folder key from preprocessing/backbone. |
| `_build_feature_transforms(preprocessing_config)`    | Preprocessing config        | `tuple[Transform, Transform]` | Builds train/eval deterministic transforms.           |
| `_sort_ascending_for_column(column, primary_metric)` | Column and metric name      | `bool`                        | Sort direction helper.                                |

---

# `modules/visualization.py`

Plotting-only utilities.

| Function                                                                                                                                                                                                                                                    | Input                                     | Output                                                    | Meaning                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| `plot_image_grid_from_df(df, n=12, path_col="path", title_col=None, subtitle_cols=None, filter_col=None, filter_value=None, sort_by=None, ascending=True, random_sample=False, seed=42, n_cols=4, figsize=None, suptitle=None, save_path=None, show=False)` | Dataframe and image-grid options          | `tuple[Figure, np.ndarray]`                               | Generic image grid for removed/wrong/extreme/sample cases. |
| `plot_sample_grid(df, n_per_class=5, path_col="path", label_col="label_name", class_order=None, seed=42, figsize=None, save_path=None, show=False)`                                                                                                         | Image dataframe                           | `tuple[Figure, np.ndarray]`                               | Plots sampled images by class.                             |
| `plot_pie_chart(df, category_col: str, title: str                                                                                                                                                                                                           | None = None, save_path=None, show=False)` | Dataframe                                                 | `tuple[Figure, Axes]`                                      | Pie chart of any categorical column's distribution. |
| `plot_bar_chart(df, category_col: str, title: str                                                                                                                                                                                                           | None = None, xlabel: str                  | None = None, ylabel="Count", save_path=None, show=False)` | Dataframe                                                  | `tuple[Figure, Axes]`                               | Bar chart of any categorical column's counts and percentages.     |
| `plot_scatter_distribution(df, x_col: str, y_col: str, hue_col: str                                                                                                                                                                                         | None = None, title: str                   | None = None, save_path=None, show=False)`                 | Dataframe                                                  | `tuple[Figure, Axes]`                               | Scatter plot for any two numeric columns (e.g., width vs height). |
| `plot_rgb_channel_kde(df, sample_per_class=300, path_col="path", label_col="label_name", class_order=None, seed=42, save_path=None, show=False)`                                                                                                            | Image dataframe                           | `tuple[Figure, np.ndarray]`                               | KDE plots of mean RGB channels by class.                   |
| `plot_metric_distribution(audit_df, metric, label_col="label_name", thresholds=None, title=None, save_path=None, show=False)`                                                                                                                               | Audit dataframe and metric                | `tuple[Figure, Axes]`                                     | Distribution plot for one audit metric.                    |
| `plot_metric_correlation_heatmap(audit_df, metrics, title="Correlation Matrix of Image Quality Metrics", save_path=None, show=False)`                                                                                                                       | Audit dataframe and metric list           | `tuple[Figure, Axes]`                                     | Correlation heatmap of selected metrics.                   |
| `plot_before_after_cleaning(summary_df, label_col="label_name", save_path=None, show=False)`                                                                                                                                                                | Cleaning summary dataframe                | `tuple[Figure, Axes]`                                     | Bar plot of before/after/removed counts.                   |
| `plot_threshold_sweep_results(sweep_df, metric_col="f1_macro", save_path=None, show=False)`                                                                                                                                                                 | Sweep result dataframe                    | `tuple[Figure, Axes]`                                     | Scatter plot of retention vs score/metric.                 |
| `plot_split_distribution(splits=None, train_df=None, val_df=None, test_df=None, label_col="label_name", save_path=None, show=False)`                                                                                                                        | Split mapping or separate split dfs       | `tuple[Figure, Axes]`                                     | Class distribution across splits.                          |
| `plot_transform_examples(transform_records, n_images=None, save_path=None, show=False)`                                                                                                                                                                     | Records with `path`, `mode`, `image`      | `tuple[Figure, np.ndarray]`                               | Shows transform preview grid.                              |
| `plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path=None, show=False)`                                                                                                                                                              | Confusion matrix dataframe or array       | `tuple[Figure, Axes]`                                     | Confusion matrix heatmap.                                  |
| `plot_grid_search_results(results_df, x="feature_extraction.backbone", y="f1_macro", hue="classifier.name", save_path=None, show=False)`                                                                                                                    | Grid result dataframe                     | `tuple[Figure, Axes]`                                     | Bar chart of grid-search results.                          |

Private helpers:

| Helper                                              | Input                  | Output                | Meaning                        |
| --------------------------------------------------- | ---------------------- | --------------------- | ------------------------------ |
| `_ensure_columns(df, columns)`                      | Dataframe and columns  | `None` or raises      | Validates required columns.    |
| `_as_numeric_series(df, column)`                    | Dataframe and column   | `pd.Series`           | Numeric conversion helper.     |
| `_read_rgb_image(path)`                             | Path                   | `Image.Image \| None` | Reads image safely.            |
| `_resolve_axes_array(axes)`                         | Matplotlib axes object | `np.ndarray`          | Flattens axes.                 |
| `_save_figure(fig, save_path, dpi=160)`             | Figure and path        | `Path \| None`        | Saves figure if path provided. |
| `_finalize_figure(fig, save_path=None, show=False)` | Figure and options     | `Figure`              | Saves/shows consistently.      |

---
