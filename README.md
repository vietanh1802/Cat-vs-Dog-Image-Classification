# Cat vs. Dog Image Classification

**Course:** Machine Learning (CO3117)
**Semester:** II — 2025–2026
**Supervisor:** ThS. Trương Vĩnh Lân
**University:** Trường Đại học Bách Khoa, ĐHQG-HCM

---

## Group Members

| # | Member | Student ID |
|---|--------|------------|
| 1 | Nguyen Manh Quoc Khanh | 2352525 |
| 2 | Phan Ngoc Lan Chi | 2352137 |
| 3 | Ngo Diem Quyen | 2353031 |
| 4 | Tran Lam Anh | 2352067 |
| 5 | Vu Duc Viet Anh | 2352074 |

---

## Objectives

This project implements a complete machine learning pipeline for binary image classification (Cat vs. Dog) on the [tongpython/cat-and-dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog) dataset. Two pipelines are developed and compared:

- **Classical Pipeline** *(required)*: Frozen pretrained CNN backbone (ResNet18, VGG16, EfficientNet-B0) for feature extraction, followed by traditional classifiers (Logistic Regression, SVM, Random Forest) with soft-voting ensemble and GridSearchCV hyperparameter tuning.
- **Deep Learning Pipeline** *(bonus)*: End-to-end two-phase fine-tuning of a pretrained CNN backbone — Phase 1 warms up the classification head while the backbone is frozen, Phase 2 fine-tunes all layers at a lower learning rate.

Both pipelines share a common preprocessing stage: EDA, data cleaning, spatial outlier removal, stratified 80/10/10 train/val/test split, letterbox resize to 224×224, and ImageNet normalization.

---

## Project Structure

```
Cat-vs-Dog-Image-Classification/
├── notebooks/
│   ├── 1_Traditional_Pipeline.ipynb    # Classical ML pipeline (frozen CNN + classifiers)
│   └── 2_Deep_Learning_Pipeline.ipynb  # End-to-end fine-tuning pipeline
├── modules/                            # Reusable Python backend library
│   ├── __init__.py
│   ├── MODULES.md                      # Full API reference for all modules
│   ├── config_utils.py                 # Configuration, seeds, device management
│   ├── config_types.py                 # Type definitions and constants
│   ├── data_utils.py                   # Dataset discovery, splits, summaries
│   ├── image_audit.py                  # Image quality metrics computation
│   ├── cleaning.py                     # Data cleaning and filtering logic
│   ├── transforms.py                   # Image preprocessing transforms
│   ├── datasets.py                     # PyTorch Dataset/DataLoader utilities
│   ├── backbones.py                    # Pretrained backbone loading and management
│   ├── feature_extraction.py           # Frozen CNN feature extraction with caching
│   ├── classical_models.py             # Classical ML classifier building and training
│   ├── deep_learning.py                # End-to-end transfer learning utilities
│   ├── evaluation.py                   # Classification metrics and reporting
│   ├── grid_search.py                  # Grid search and hyperparameter tuning
│   ├── threshold_experiments.py        # Sweet Spot threshold experiments
│   ├── artifacts.py                    # Model/data saving and loading
│   └── visualization.py               # Plotting utilities
├── features/                           # Cached extracted features (.npy files)
│   ├── augmented_224_efficientnet_b0/  # EfficientNet-B0 features (1280-dim)
│   ├── augmented_224_resnet18/         # ResNet18 features (512-dim)
│   └── augmented_224_vgg16/            # VGG16 features (512-dim)
└── reports/
    └── Report_Group5.pdf               # Final project report
```

---

## Dataset

- **Source:** [tongpython/cat-and-dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog) on Kaggle
- **Total images:** 10,028 (5,011 cats / 5,017 dogs) — nearly balanced
- **After spatial outlier removal (3-sigma):** 10,002 images (26 removed)
- **Final split (stratified):**

| Split | Total | Cat | Dog |
|-------|-------|-----|-----|
| Train | 8,001 | 3,998 | 4,003 |
| Validation | 1,000 | 500 | 500 |
| Test | 1,001 | 500 | 501 |

- **Image size range:** width 57–1,050 px, height 33–768 px
- **Preprocessing:** Letterbox resize to 224×224, normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## Key EDA Findings

| Finding | Evidence | Pipeline Decision |
|---------|----------|-------------------|
| Balanced classes | ~50% cat / ~50% dog | No class weighting needed |
| High resolution variability | Width std ~108 px | Letterbox resize to 224×224 |
| Blur is dominant quality issue | Laplacian variance < 40 for ~2–3% | Optional blur threshold cleaning |
| No colour discriminability | RGB KDE strongly overlaps between classes | Rely on shape/texture features via CNN |
| Near-zero missing values | 0 nulls in metadata | No imputation required |

---

## How to Run

### Google Colab (Recommended)

1. Open one of the notebooks in **Google Colab**:
   - `notebooks/1_Traditional_Pipeline.ipynb` — Classical ML pipeline
   - `notebooks/2_Deep_Learning_Pipeline.ipynb` — Deep Learning pipeline
2. Go to **Runtime → Run all**
3. No manual setup required — each notebook will:
   - Download the dataset automatically via `kagglehub`
   - Install any additional libraries if needed
   - Run the full pipeline end-to-end

> **Note (Classical Pipeline):** Feature extraction from 3 backbones may take 20–40 minutes on first run. Extracted features are cached to `./features/` as `.npy` files and reused automatically on subsequent runs.

> **Note (Deep Learning Pipeline):** Full fine-tuning requires a GPU runtime. Enable it in Colab via **Runtime → Change runtime type → GPU**. Training takes approximately 30 minutes.

---

## Results

### Classical Pipeline

| Backbone | Feature Dim | Classifier | Test Accuracy | F1 Macro | ROC-AUC |
|----------|-------------|------------|:-------------:|:--------:|:-------:|
| EfficientNet-B0 | 1280 | Logistic Regression (C=0.1) | **99.4%** | **99.4%** | **0.9999** |
| ResNet18 | 512 | Logistic Regression | ~98.5% | ~98.5% | ~0.999 |
| VGG16 | 512 | Logistic Regression | ~98.2% | ~98.2% | ~0.999 |

Best model: **EfficientNet-B0 + Logistic Regression** — 6 wrong predictions out of 1,003 test samples.

### Deep Learning Pipeline

| Backbone | Parameters | Test Accuracy | F1 Macro | ROC-AUC | Epochs |
|----------|-----------|:-------------:|:--------:|:-------:|:------:|
| EfficientNet-B0 (fine-tuned) | 4.01M | **98.7%** | **98.7%** | **0.9992** | 19 (early stopped) |

Best validation loss: **0.2273** at epoch 14. 13 wrong predictions out of 1,001 test samples.

### Pipeline Comparison

| Aspect | Classical | Deep Learning |
|--------|-----------|---------------|
| **Feature source** | Frozen pretrained CNN | End-to-end fine-tuning |
| **Training time** | ~77s (extract) + ~6s (classifier) | ~30 min on GPU |
| **Test accuracy** | **99.4%** | 98.7% |
| **ROC-AUC** | **0.9999** | 0.9992 |
| **Interpretability** | High (Logistic Regression coefficients) | Lower (learned representations) |
| **Computational cost** | Low — CPU-friendly | High — GPU required |
| **Data dependency** | Works well with moderate data | Benefits from larger datasets |

> The Classical pipeline slightly outperforms the Deep Learning pipeline on this dataset, likely because the pretrained ImageNet features are already highly discriminative for cats vs. dogs. Fine-tuning adds flexibility but is overkill at this scale.

---

## Pipeline Details

### Classical Pipeline (`1_Traditional_Pipeline.ipynb`)

1. **EDA** — class balance, spatial distribution, RGB distributions, image quality audit (blur, entropy, saturation, saliency)
2. **Preprocessing** — optional quality-based cleaning, stratified split, transform preview
3. **Feature extraction** — frozen backbone forward pass on all splits, cached as `.npy`
4. **Training** — baseline classifiers → GridSearchCV (CV=3) → best model by F1 macro
5. **Evaluation** — per-split metrics, classification report, confusion matrix, misclassified image gallery

### Deep Learning Pipeline (`2_Deep_Learning_Pipeline.ipynb`)

1. **EDA** — class distribution, spatial scatter, quality metrics, RGB KDE by class
2. **Preprocessing** — cleaning, spatial outlier removal (3-sigma), stratified split, augmentation preview (RandomResizedCrop, RandomRotation, ColorJitter)
3. **Model** — pretrained EfficientNet-B0 with fresh 2-class classification head
4. **Training**
   - Phase 1 (warm-up): 3 epochs, backbone frozen, LR = 1e-4
   - Phase 2 (fine-tuning): up to 17 epochs, all layers unfrozen, LR = 1e-5
   - Loss: CrossEntropyLoss with label smoothing = 0.1
   - Optimizer: AdamW with weight decay; Scheduler: CosineAnnealingLR; Early stopping patience = 5
5. **Evaluation** — classification report, confusion matrix, ROC curve, misclassified image gallery, **Grad-CAM** visualization

---

## Modules

The `modules/` directory provides a reusable Python backend library. Notebooks act as a front-end (explanation, configuration, execution, visualization) while modules provide clean, testable back-end logic.

| Module | Purpose |
|--------|---------|
| `config_utils` | Config loading, seed setting, device selection, path resolution |
| `config_types` | Type definitions and validation constants |
| `data_utils` | Dataset discovery, dataframe creation, stratified splitting |
| `image_audit` | Image quality metrics (blur, entropy, saturation, saliency) |
| `cleaning` | Quality-based filtering, duplicate removal, mask building |
| `transforms` | Image preprocessing (letterbox resize, augmentation, normalization) |
| `datasets` | PyTorch `Dataset`/`DataLoader` implementations |
| `backbones` | Pretrained model loading and classification head replacement |
| `feature_extraction` | Frozen CNN feature extraction with `.npy` caching |
| `classical_models` | Sklearn classifiers, parameter grids, grid search orchestration |
| `deep_learning` | Transfer learning utilities, training loops, checkpointing |
| `evaluation` | Metrics, reports, confusion matrices, misclassification analysis |
| `grid_search` | Grid search orchestration for classical pipeline |
| `threshold_experiments` | Sweet Spot threshold evaluation for data cleaning |
| `artifacts` | JSON/pickle/CSV/NumPy save and load utilities |
| `visualization` | Plotting functions (grids, distributions, heatmaps, curves, Grad-CAM) |

See [modules/MODULES.md](modules/MODULES.md) for the full API reference.

---

## Links

- [Report PDF](reports/Report_Group5.pdf)
- [Dataset — tongpython/cat-and-dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
