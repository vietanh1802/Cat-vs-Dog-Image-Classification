# Cat vs. Dog Image Classification

**Course:** Machine Learning (CO3117)
**Semester:** II — 2025–2026
**Supervisor:** ThS. Trương Vĩnh Lân
**University:** Trường Đại học Bách Khoa, ĐHQG-HCM

---

## 👥 Group Members

| # | Member | Student ID |
|---|--------|------------|
| 1 | Nguyen Manh Quoc Khanh | 2352525 |
| 2 | Phan Ngoc Lan Chi | 2352137 |
| 3 | Ngo Diem Quyen | 2353031 |
| 4 | Tran Lam Anh | 2352067 |
| 5 | Vu Duc Viet Anh | 2352074 |

---

## 🎯 Objectives

This project implements a complete machine learning pipeline for binary image classification (Cat vs. Dog) on the [tongpython/cat-and-dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog) dataset. Two pipelines are developed and compared:

- **Classical Pipeline** *(required)*: frozen pretrained CNN backbone (ResNet18, VGG16, EfficientNet-B0) for feature extraction, followed by traditional classifiers (Logistic Regression, SVM, Random Forest) with soft-voting ensemble.
- **Deep Learning Pipeline** *(bonus)*: end-to-end fine-tuning of a pretrained CNN backbone with a new classification head, trained with CrossEntropyLoss and Adam optimizer.

Both pipelines share a common EDA and preprocessing stage (cleaning, outlier removal, stratified split, letterbox resize to 224×224, normalisation to [0, 1]).

---

## 🗂️ Project Structure

```
Cat-vs-Dog-Image-Classification/
├── notebooks/
│   └── main.ipynb          # Main Colab notebook (Run all to reproduce)
├── modules/
│   └── image_preprocessor.py
├── reports/
│   └── report.pdf
└── features/               # Extracted .npy feature files (classical pipeline)
```

---

## ▶️ How to Run

1. Open `notebooks/main.ipynb` in **Google Colab**
2. Go to **Runtime → Run all**
3. No manual setup required — the notebook will:
   - Download the dataset automatically via `kagglehub`
   - Install any additional libraries if needed
   - Run the full pipeline end-to-end

> **Note:** The Classical Pipeline extracts features from 3 backbones and may take 20–40 minutes on first run. Extracted features are cached to `./features/classical/` and reused on subsequent runs.

---

## 📊 Results

| Pipeline | Backbone | Classifier | Test Accuracy |
|----------|----------|------------|---------------|
| Classical | VGG16 | Logistic Regression | — |
| Classical | ResNet18 | Random Forest | — |
| Deep Learning | ResNet18 | Fine-tuned head | — |

> Results will be updated after final evaluation.

---

## 🔗 Links

- 📓 [Colab Notebook](#) *(update link here)*
- 📄 [Report PDF](#) *(update link here)*
- 📦 [Dataset — tongpython/cat-and-dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
