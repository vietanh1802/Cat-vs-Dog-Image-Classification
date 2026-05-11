# Tài liệu ôn tập phỏng vấn — Cat vs. Dog Image Classification

---

## Mục lục

1. [Machine Learning cơ bản](#1-machine-learning-cơ-bản)
2. [Mạng nơ-ron tích chập (CNN)](#2-mạng-nơ-ron-tích-chập-cnn)
3. [Transfer Learning & Fine-tuning](#3-transfer-learning--fine-tuning)
4. [Các backbone được dùng trong dự án](#4-các-backbone-được-dùng-trong-dự-án)
5. [Classical ML Classifiers](#5-classical-ml-classifiers)
6. [Tiền xử lý dữ liệu & Data Augmentation](#6-tiền-xử-lý-dữ-liệu--data-augmentation)
7. [Đánh giá mô hình](#7-đánh-giá-mô-hình)
8. [Tối ưu hóa & Regularization](#8-tối-ưu-hóa--regularization)
9. [Pipeline Classical vs. Deep Learning — So sánh & Phân tích](#9-pipeline-classical-vs-deep-learning--so-sánh--phân-tích)
10. [Câu hỏi phỏng vấn thường gặp kèm trả lời mẫu](#10-câu-hỏi-phỏng-vấn-thường-gặp-kèm-trả-lời-mẫu)

---

## 1. Machine Learning cơ bản

### 1.1 Supervised Learning

Supervised learning là bài toán học từ tập dữ liệu có nhãn `(X, y)`, mục tiêu là học hàm `f: X → y`.

- **Phân loại (Classification):** đầu ra là nhãn rời rạc. Bài toán này là binary classification (Cat = 0, Dog = 1).
- **Hồi quy (Regression):** đầu ra là giá trị liên tục.

### 1.2 Bias–Variance Tradeoff

| | Bias cao | Variance cao |
|--|----------|-------------|
| **Biểu hiện** | Underfitting — model quá đơn giản | Overfitting — model ghi nhớ training data |
| **Xử lý** | Tăng độ phức tạp model, thêm features | Regularization, dropout, thu thập thêm data |

**Công thức:** `Expected Error = Bias² + Variance + Irreducible Noise`

### 1.3 Chiến lược chia dữ liệu

Dự án dùng **Stratified Split 80/10/10**:
- **Stratified:** tỉ lệ class (cat/dog) được bảo toàn trong từng split, tránh tình huống một split bị lệch class.
- **Train (80%):** học tham số mô hình.
- **Validation (10%):** chọn hyperparameter, quyết định early stopping.
- **Test (10%):** đánh giá cuối cùng, **không được dùng để điều chỉnh mô hình**.

> **Câu hỏi hay:** Tại sao không dùng toàn bộ dữ liệu để train? — Vì ta cần ước lượng khả năng generalize của mô hình trên dữ liệu chưa thấy; nếu train trên toàn bộ data thì không có điểm tham chiếu khách quan.

### 1.4 Cross-Validation

Dự án dùng **GridSearchCV với CV=3** (3-fold cross-validation) cho hyperparameter tuning:
- Chia training set thành 3 phần bằng nhau.
- Mỗi vòng, 2 phần dùng để train, 1 phần dùng để validate.
- Lấy trung bình kết quả 3 vòng → ổn định hơn so với chỉ dùng 1 validation set.

---

## 2. Mạng nơ-ron tích chập (CNN)

### 2.1 Tại sao CNN hiệu quả với ảnh?

Ảnh có **spatial locality** (điểm ảnh gần nhau có tương quan cao) và **translation invariance** (một con mèo ở góc trái hay góc phải vẫn là mèo). CNN khai thác hai tính chất này thông qua:

- **Convolutional layer:** kernel trượt trên ảnh, chia sẻ trọng số (`weight sharing`) → ít tham số hơn fully-connected.
- **Pooling layer (MaxPool/AvgPool):** giảm chiều không gian, tăng receptive field, tạo bất biến dịch chuyển.
- **Receptive field:** vùng ảnh gốc ảnh hưởng đến một neuron đầu ra; layers sâu hơn có receptive field lớn hơn → học đặc trưng toàn cục (hình dạng, cấu trúc).

### 2.2 Các thành phần của CNN

```
Input → [Conv → BN → ReLU → Pool] × N → Flatten → FC → Output
```

| Thành phần | Vai trò |
|-----------|---------|
| Convolution | Trích xuất đặc trưng cục bộ (edge, texture, pattern) |
| Batch Normalization | Chuẩn hóa activation, ổn định training, cho phép lr cao hơn |
| ReLU | Hàm kích hoạt phi tuyến, giải quyết vanishing gradient |
| MaxPooling | Giảm kích thước feature map, giữ lại đặc trưng nổi bật nhất |
| Dropout | Regularization — tắt ngẫu nhiên neurons khi train |
| FC Layer | Tổng hợp đặc trưng, phân loại cuối |

### 2.3 Feature Hierarchy trong CNN

- **Layers đầu (shallow):** phát hiện đặc trưng cấp thấp — cạnh (edges), góc, gradient màu sắc.
- **Layers giữa (mid):** kết hợp thành texture, pattern, bộ phận cơ thể.
- **Layers cuối (deep):** đặc trưng cấp cao — mặt mèo, tai chó, lông, v.v.

Đây là lý do tại sao **frozen pretrained CNN vẫn cho features rất tốt** cho bài toán này — những đặc trưng cấp cao đã được học trên ImageNet rất có liên quan đến việc phân biệt mèo và chó.

### 2.4 Vanishing / Exploding Gradient

- **Vanishing gradient:** gradient quá nhỏ ở các layer đầu → layers đầu học rất chậm. Giải pháp: ReLU, Batch Norm, ResNet skip connections.
- **Exploding gradient:** gradient quá lớn → weights phát散. Giải pháp: gradient clipping, weight initialization tốt.

---

## 3. Transfer Learning & Fine-tuning

### 3.1 Transfer Learning là gì?

Transfer learning là tái sử dụng kiến thức (trọng số) đã học từ bài toán nguồn (thường là ImageNet-1k với 1.28 triệu ảnh, 1000 class) để áp dụng cho bài toán đích (Cat vs. Dog).

**Lý do hiệu quả:**
- Đặc trưng học được trên ImageNet (edge, texture, shape) mang tính tổng quát cao.
- Dataset đích thường nhỏ hơn nhiều; train từ đầu sẽ overfit.
- Tiết kiệm thời gian và tài nguyên tính toán.

### 3.2 Hai chiến lược trong dự án

#### Chiến lược 1 — Feature Extraction (Classical Pipeline)

```
Pretrained CNN (FROZEN) → Feature Vector → Classical Classifier
```

- Backbone bị **freeze hoàn toàn** — không cập nhật gradient.
- Output của Global Average Pooling được dùng làm feature vector (1280-dim với EfficientNet-B0).
- Classical classifier (Logistic Regression, SVM, ...) được train trên các feature vector này.
- **Ưu điểm:** Cực kỳ nhanh, ổn định, không cần GPU cho phase classifier.
- **Nhược điểm:** Feature không được điều chỉnh cho bài toán đích.

#### Chiến lược 2 — Fine-tuning (Deep Learning Pipeline)

```
Phase 1 (warm-up): Backbone FROZEN → train head (3 epochs, LR=1e-4)
Phase 2 (fine-tune): Backbone UNFROZEN → train all layers (≤17 epochs, LR=1e-5)
```

**Tại sao cần warm-up trước?**
Nếu mở toàn bộ backbone ngay từ đầu trong khi head có weights ngẫu nhiên, gradient lớn từ head sẽ "phá" trọng số pretrained của backbone. Warm-up giúp head hội tụ trước, sau đó fine-tune toàn bộ với LR nhỏ hơn để tinh chỉnh nhẹ.

**Tại sao LR phase 2 nhỏ hơn phase 1?**
Backbone đã có weights tốt từ pretrained. LR nhỏ (1e-5) giúp điều chỉnh nhẹ nhàng, tránh "quên" (catastrophic forgetting) kiến thức đã học.

### 3.3 Catastrophic Forgetting

Khi fine-tune toàn bộ mô hình với LR quá lớn, mô hình sẽ "quên" các đặc trưng tổng quát đã học từ ImageNet và overfit vào dataset đích nhỏ. Giải pháp:
- Dùng LR nhỏ cho backbone.
- Có thể dùng **differential learning rates** (LR khác nhau cho từng layer group).

---

## 4. Các backbone được dùng trong dự án

### 4.1 ResNet18

- **Năm:** 2015 (He et al., CVPR 2016)
- **Kiến trúc:** 18 layers với **Residual Connections** (skip connections)
- **Feature dim:** 512

**Skip connection:** `output = F(x) + x`

Giải quyết vanishing gradient — gradient có thể đi thẳng qua skip connection về layers đầu. Cho phép train mạng rất sâu (ResNet-152, ResNet-1000+).

### 4.2 VGG16

- **Năm:** 2014 (Simonyan & Zisserman, ICLR 2015)
- **Kiến trúc:** 16 layers, chỉ dùng 3×3 Conv, rất nhiều tham số (~138M)
- **Feature dim:** 512

**Đặc điểm:** Đơn giản, uniformly deep, nhưng nặng. Hiện ít dùng do hiệu quả tham số kém hơn ResNet và EfficientNet.

### 4.3 EfficientNet-B0 *(Best performer trong dự án)*

- **Năm:** 2019 (Tan & Le, ICML 2019)
- **Kiến trúc:** Compound Scaling — tỉ lệ đồng thời depth, width, resolution.
- **Feature dim:** 1280 (Global Average Pooling output)
- **Tham số:** ~4M (nhỏ hơn VGG16 hơn 30 lần)

**Mobile Inverted Bottleneck (MBConv):** block chính của EfficientNet
```
Input → Expand (pointwise 1×1) → Depthwise 3×3 → SE block → Project (pointwise 1×1) → Skip
```

**Squeeze-and-Excitation (SE) block:** học channel attention — "kênh màu nào quan trọng nhất?" cho từng vị trí không gian.

**Tại sao EfficientNet-B0 tốt nhất?**
- Feature dim 1280 > ResNet18 (512) → thông tin phong phú hơn.
- SE attention chọn lọc kênh quan trọng → features phân biệt hơn.
- Compound scaling cân bằng tốt depth/width/resolution.

---

## 5. Classical ML Classifiers

### 5.1 Logistic Regression

**Hàm dự đoán:** `P(y=1|x) = σ(wᵀx + b)` với `σ(z) = 1/(1+e⁻ᶻ)`

**Hàm mất mát:** Binary Cross-Entropy (Log Loss)
```
L = -[y·log(p) + (1-y)·log(1-p)]
```

**Hyperparameter C (Regularization strength):**
- C lớn → regularization yếu → có thể overfit.
- C nhỏ → regularization mạnh → mô hình đơn giản hơn.
- Dự án chọn **C=0.1** qua GridSearchCV → regularization vừa phải.

**Tại sao Logistic Regression lại tốt nhất?**
Features từ EfficientNet-B0 (1280-dim) đã rất phân biệt → bài toán gần như **linearly separable** trong không gian feature. Logistic Regression — vốn là linear classifier — phù hợp hoàn hảo.

### 5.2 Support Vector Machine (SVM)

**Ý tưởng:** Tìm hyperplane có **margin lớn nhất** giữa hai class.

```
max margin = 2/||w||  subject to  yᵢ(wᵀxᵢ + b) ≥ 1
```

**Kernel trick:** Chiếu dữ liệu vào không gian chiều cao hơn mà không cần tính toán tường minh.
- `Linear kernel`: K(x,z) = xᵀz — thích hợp khi data linearly separable (như bài toán này).
- `RBF kernel`: K(x,z) = exp(-γ||x-z||²) — thích hợp khi data không linear.

**Hyperparameters:** C (soft margin trade-off), γ (RBF kernel width)

**Support vectors:** Chỉ các điểm gần margin nhất ảnh hưởng đến hyperplane — SVM không nhạy cảm với outliers xa margin.

### 5.3 Random Forest

**Nguyên lý:** Ensemble của N decision trees (dự án thử N=100, 200).

**Hai nguồn randomness:**
1. **Bootstrap sampling (Bagging):** Mỗi tree được train trên một tập con ngẫu nhiên (với thay thế) của training data.
2. **Feature subsampling:** Tại mỗi split, chỉ chọn ngẫu nhiên `sqrt(d)` features → giảm correlation giữa các trees.

**Kết quả:** Variance giảm nhiều (ensemble), bias không tăng đáng kể.

**Feature importance:** Tính bằng trung bình lượng giảm impurity (Gini/entropy) qua tất cả trees — cho biết feature nào quan trọng nhất.

### 5.4 Ensemble — Soft Voting

```python
final_prob = (1/N) * Σ pᵢ(x)   # trung bình xác suất từ N classifiers
```

Dùng **probability averaging** (soft voting) thay vì majority voting → ổn định hơn vì tận dụng được confidence của từng model.

### 5.5 GridSearchCV

- Thử tất cả tổ hợp hyperparameter trong grid đã định nghĩa.
- Với mỗi tổ hợp, đánh giá bằng k-fold cross-validation.
- Chọn tổ hợp tốt nhất theo metric chỉ định (F1 macro trong dự án).

**Scoring F1 macro:** Tính F1 độc lập trên từng class, lấy trung bình không trọng số → phù hợp với balanced dataset.

---

## 6. Tiền xử lý dữ liệu & Data Augmentation

### 6.1 LetterBox Resize

```
Ảnh gốc (W×H) → Scale để fit vào 224×224 (giữ aspect ratio) → Pad phần còn lại bằng màu xám (128,128,128)
```

**Tại sao không resize thẳng về 224×224?**
Resize trực tiếp làm méo ảnh → thay đổi tỉ lệ tương quan của đối tượng → ảnh hưởng đến đặc trưng hình dạng mà CNN học được.

### 6.2 ImageNet Normalization

```python
mean = [0.485, 0.456, 0.406]  # RGB mean của ImageNet
std  = [0.229, 0.224, 0.225]  # RGB std của ImageNet
pixel_normalized = (pixel/255 - mean) / std
```

**Lý do:** Backbone được pretrain với ImageNet normalization. Nếu không chuẩn hóa đúng, distribution của input sẽ khác với lúc pretrain → features không còn ý nghĩa.

### 6.3 Data Augmentation (Deep Learning Pipeline)

Augmentation chỉ áp dụng cho **training set**, không áp dụng cho val/test.

| Augmentation | Tham số | Lý do |
|-------------|---------|-------|
| RandomResizedCrop(224, scale=(0.8,1.0)) | Crop 80-100% diện tích | Bất biến vị trí đối tượng |
| RandomHorizontalFlip(p=0.5) | 50% | Bất biến lật ngang |
| RandomRotation(15°) | ±15° | Bất biến góc xoay nhỏ |
| ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) | | Bất biến điều kiện ánh sáng |

**Mục đích:** Tăng effective dataset size, giảm overfitting, học được representations bất biến với các biến đổi không liên quan.

### 6.4 Data Cleaning

Dự án thực hiện các bước kiểm tra chất lượng ảnh:

| Metric | Công thức | Ngưỡng loại |
|--------|-----------|-------------|
| **Blur** | Laplacian variance: `var(∇²I)` | < 40 → quá mờ |
| **Entropy** | `H = -Σ p·log(p)` trên histogram | < 3.0 → ít thông tin |
| **Saturation** | Mean saturation trong HSV | < 0.05 → gần grayscale |
| **Saliency** | Sobel edge magnitude ở trung tâm | < 0.02 → không có đối tượng rõ ràng |
| **Near-duplicate** | Perceptual hashing | Hamming distance < 5 → giữ 1 |

**Spatial outlier removal:** Loại bỏ ảnh có width hoặc height lệch hơn 3 sigma so với phân phối — 26 ảnh bị loại từ 10,028.

---

## 7. Đánh giá mô hình

### 7.1 Confusion Matrix

```
                Predicted Dog    Predicted Cat
Actual Dog    |   TP           |   FN          |
Actual Cat    |   FP           |   TN          |
```

- **TP (True Positive):** Dự đoán đúng là Dog.
- **TN (True Negative):** Dự đoán đúng là Cat.
- **FP (False Positive):** Dự đoán sai Dog khi thực tế là Cat (Type I Error).
- **FN (False Negative):** Dự đoán sai Cat khi thực tế là Dog (Type II Error).

### 7.2 Các metrics quan trọng

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Tỉ lệ đúng tổng thể |
| **Precision** | TP/(TP+FP) | Trong những gì model dự đoán là Positive, bao nhiêu đúng? |
| **Recall (Sensitivity)** | TP/(TP+FN) | Trong những Positive thực sự, model tìm được bao nhiêu? |
| **F1 Score** | 2·P·R/(P+R) | Harmonic mean của Precision và Recall |
| **Specificity** | TN/(TN+FP) | Tỉ lệ Negative được phát hiện đúng |

**Khi nào dùng metric nào?**
- Dataset imbalanced → Accuracy bị misleading, dùng F1.
- FN tốn kém (bệnh hiểm không phát hiện) → ưu tiên Recall.
- FP tốn kém (spam filter nhầm email quan trọng) → ưu tiên Precision.
- Dataset balanced như dự án này → cả hai metric đều phù hợp.

### 7.3 ROC Curve & AUC

**ROC (Receiver Operating Characteristic):** Đồ thị TPR (Recall) vs. FPR (1-Specificity) khi thay đổi ngưỡng phân loại (threshold) từ 0 → 1.

**AUC (Area Under Curve):**
- AUC = 1.0: Perfect classifier.
- AUC = 0.5: Classifier ngẫu nhiên (đường chéo).
- AUC của dự án: **0.9999** (Classical) và **0.9992** (Deep Learning) → gần hoàn hảo.

**Ý nghĩa AUC:** Xác suất mà model xếp hạng một mẫu Positive ngẫu nhiên cao hơn một mẫu Negative ngẫu nhiên.

### 7.4 Macro vs. Weighted Averaging

- **Macro:** Tính metric độc lập cho từng class, lấy trung bình không trọng số → ưu tiên bình đẳng giữa các class.
- **Weighted:** Trung bình có trọng số theo số mẫu của từng class → phù hợp với imbalanced dataset.
- Dự án dùng **Macro** vì dataset cân bằng (~50%/50%).

### 7.5 Grad-CAM (Gradient-weighted Class Activation Mapping)

Kỹ thuật visualization cho phép **hiểu mô hình DL đang nhìn vào đâu** khi đưa ra quyết định:

```
1. Forward pass → tính class score
2. Backward pass → gradient của score đối với feature map ở layer cuối cùng
3. Global Average Pooling các gradient → trọng số α cho từng channel
4. Weighted sum các feature map: L = ReLU(Σ αₖ · Aᵏ)
5. Upsample heatmap về kích thước ảnh gốc
```

Trong dự án, Grad-CAM cho thấy mô hình tập trung vào **khuôn mặt và tai** của chó/mèo — xác nhận mô hình học đúng đặc trưng.

---

## 8. Tối ưu hóa & Regularization

### 8.1 Adam & AdamW Optimizer

**Adam** kết hợp Momentum và RMSProp:
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          # moment bậc 1 (gradient mean)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         # moment bậc 2 (gradient variance)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)  # update
```

Mặc định: β₁=0.9, β₂=0.999, ε=1e-8

**AdamW:** Adam + decoupled weight decay
```
θ_t = θ_{t-1} - α·(m̂_t/(√v̂_t + ε) + λ·θ_{t-1})
```

Weight decay trong AdamW tách biệt khỏi gradient update → regularization hiệu quả hơn Adam thông thường.

### 8.2 Learning Rate Scheduler — CosineAnnealingLR

```
LR_t = LR_min + 0.5·(LR_max - LR_min)·(1 + cos(π·t/T))
```

LR giảm dần theo hình dạng cosine từ LR_max về LR_min qua T epochs. Dự án dùng riêng cho mỗi phase training. Ưu điểm: LR giảm mượt mà, tránh dao động cuối khi gần hội tụ.

### 8.3 Early Stopping

Dừng training khi validation loss không cải thiện sau `patience=5` epochs liên tiếp:
- Tránh overfitting.
- Tiết kiệm thời gian.
- Trong dự án: dừng tại epoch 19 (best val loss tại epoch 14).

### 8.4 Label Smoothing

Thay vì one-hot labels `[1, 0]`, dùng `[1-ε, ε]` với `ε=0.1`:
```
y_smooth = (1-ε)·y_one_hot + ε/K   # K = số class
```

**Lý do:** Ngăn mô hình trở nên quá tự tin (overconfident), giảm overfitting, cải thiện calibration. Thích hợp khi nhãn có thể có nhiễu (một số ảnh mèo/chó mơ hồ).

### 8.5 Batch Normalization

Chuẩn hóa activation trong mỗi mini-batch:
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γ·x̂ + β   # γ, β là tham số học được
```

**Lợi ích:**
- Cho phép dùng learning rate lớn hơn.
- Giảm nhạy cảm với weight initialization.
- Có tác dụng regularization nhẹ.
- Giúp gradient không bị vanish/explode.

### 8.6 Dropout

Tắt ngẫu nhiên p% neurons khi training:
- Ngăn co-adaptation — các neurons buộc phải học độc lập.
- Tương đương training ensemble của 2^N sub-networks.
- Khi inference: không dropout nhưng scale output × (1-p) [hoặc dùng inverted dropout].

### 8.7 Mixed Precision Training (AMP)

Dự án dùng `use_amp=True` với `torch.cuda.amp`:
- Tính toán với **float16** (nhanh hơn, ít bộ nhớ hơn) nhưng accumulate gradient với **float32** (ổn định hơn).
- Giảm memory GPU ~40% → tăng được batch size.
- Cần `GradScaler` để tránh gradient underflow với float16.

---

## 9. Pipeline Classical vs. Deep Learning — So sánh & Phân tích

### 9.1 Kết quả cuối cùng

| | Classical (EfficientNet-B0 + LR) | Deep Learning (Fine-tuned EfficientNet-B0) |
|--|:---:|:---:|
| **Test Accuracy** | **99.4%** | 98.7% |
| **F1 Macro** | **99.4%** | 98.7% |
| **ROC-AUC** | **0.9999** | 0.9992 |
| **Sai** | 6/1003 ảnh | 13/1001 ảnh |
| **Training time** | ~83s (CPU) | ~30 phút (GPU) |

### 9.2 Tại sao Classical lại tốt hơn trong bài toán này?

1. **Dataset size nhỏ vừa phải (~10k ảnh):** Fine-tuning toàn bộ backbone (~4M params) với dataset nhỏ dễ overfit hơn.
2. **ImageNet features đã rất tốt:** Mèo và chó đều xuất hiện trong ImageNet → backbone đã học đặc trưng phân biệt.
3. **Logistic Regression là linear classifier:** Khi data đã linearly separable trong feature space, linear classifier = optimal. Fine-tuning thêm không cải thiện.
4. **Augmentation tradeoff:** Classical pipeline sử dụng augmentation ở extraction nhưng features đã ổn định; DL pipeline cần augmentation mạnh hơn.

### 9.3 Khi nào nên dùng Deep Learning?

- Dataset lớn (>100k ảnh).
- Bài toán domain-specific (y tế, vệ tinh) — pretrained features kém liên quan.
- Cần học đặc trưng cực kỳ cụ thể cho task.
- Có GPU mạnh và thời gian training không phải vấn đề.

### 9.4 Khi nào nên dùng Classical + Feature Extraction?

- Dataset nhỏ đến vừa.
- Tài nguyên tính toán hạn chế.
- Cần kết quả nhanh và interpretable.
- ImageNet features đã liên quan đến bài toán.

---

## 10. Câu hỏi phỏng vấn thường gặp kèm trả lời mẫu

### Q1: Giải thích pipeline của bạn từ đầu đến cuối.

**Trả lời:**
> "Dự án triển khai hai pipeline song song. Pipeline thứ nhất (Classical): tôi dùng EfficientNet-B0 pretrained trên ImageNet như một feature extractor cố định — toàn bộ backbone bị freeze, chỉ forward pass để lấy 1280-dim feature vector từ Global Average Pooling. Sau đó train Logistic Regression với L2 regularization (C=0.1 chọn qua GridSearchCV với 3-fold CV) trên các features này. Pipeline thứ hai (Deep Learning): tôi fine-tune EfficientNet-B0 theo hai phase — warm-up 3 epochs với backbone frozen và LR=1e-4, rồi fine-tune toàn bộ với LR=1e-5. Cả hai pipeline đều dùng chung preprocessing: letterbox resize về 224×224, ImageNet normalization, và stratified 80/10/10 split. Kết quả: Classical đạt 99.4% accuracy, Deep Learning đạt 98.7%."

---

### Q2: Tại sao EfficientNet-B0 tốt hơn ResNet18 và VGG16 trong bài toán này?

**Trả lời:**
> "Ba lý do chính: (1) Feature dimension — EfficientNet-B0 cho 1280-dim features trong khi ResNet18 và VGG16 chỉ cho 512-dim, thông tin phong phú hơn cho classifier phân biệt. (2) Squeeze-and-Excitation block trong EfficientNet học channel attention — mô hình biết channel nào quan trọng nhất, tạo ra features chọn lọc hơn. (3) Compound scaling cân bằng tốt depth, width và resolution — không trade-off một chiều như ResNet (depth) hay VGG (chỉ depth đơn giản)."

---

### Q3: Tại sao dùng Logistic Regression mà không phải Neural Network cho Classical Pipeline?

**Trả lời:**
> "Khi features từ EfficientNet-B0 đã ở chiều cao (1280-dim) và chứa thông tin phân biệt phong phú, bài toán trở nên gần như linearly separable trong không gian feature đó. Logistic Regression — vốn là linear classifier — phù hợp hoàn hảo với bài toán linearly separable, ít tham số, không overfit, và training cực nhanh. Kết quả 99.4% trên test set (chỉ 6 ảnh sai) chứng minh điều này. Một Neural Network sẽ phức tạp hơn không cần thiết và có nguy cơ overfit với dataset này."

---

### Q4: Giải thích Grad-CAM và tại sao nó quan trọng?

**Trả lời:**
> "Grad-CAM là kỹ thuật tạo heatmap hiển thị những vùng ảnh mà model DL đang tập trung khi đưa ra dự đoán. Cụ thể: sau forward pass tính class score, ta backward để lấy gradient của score đối với feature map của convolutional layer cuối. Gradient được Global Average Pooling để tính trọng số cho từng channel, rồi weighted sum và upsample về kích thước ảnh. Nó quan trọng vì: thứ nhất, xác nhận model học đúng đặc trưng (trong dự án, model nhìn vào mặt và tai của mèo/chó chứ không phải background). Thứ hai, giúp debug khi model sai — nếu model nhìn vào sai vùng thì biết cần thêm augmentation hay clean data."

---

### Q5: Overfitting là gì? Làm thế nào để phát hiện và xử lý?

**Trả lời:**
> "Overfitting xảy ra khi model ghi nhớ training data thay vì học pattern tổng quát — training accuracy cao nhưng validation/test accuracy thấp hơn đáng kể. Phát hiện: so sánh training loss và validation loss theo epoch — nếu training loss tiếp tục giảm nhưng val loss tăng hoặc dừng giảm, đó là overfitting. Xử lý: (1) Regularization — L2 trong Logistic Regression (tham số C), weight decay trong AdamW; (2) Dropout; (3) Data augmentation — tăng effective dataset size; (4) Early stopping — dừng khi val loss không cải thiện; (5) Label smoothing — ngăn model quá tự tin; (6) Thu thập thêm dữ liệu — phương pháp hiệu quả nhất."

---

### Q6: Stratified split khác gì với random split?

**Trả lời:**
> "Random split chia dữ liệu ngẫu nhiên — có thể xảy ra tình huống split có tỉ lệ class lệch so với phân phối gốc, đặc biệt nguy hiểm với imbalanced dataset hoặc dataset nhỏ. Stratified split đảm bảo mỗi split giữ nguyên tỉ lệ class như phân phối gốc. Ví dụ, nếu dataset có 50% cat/50% dog, thì train, val, test đều phải có ~50%/50%. Dự án dùng stratified split để đảm bảo kết quả đánh giá không bị bias bởi phân phối class không đồng đều trong test set."

---

### Q7: Batch Normalization hoạt động như thế nào và tại sao lại hiệu quả?

**Trả lời:**
> "Batch Normalization chuẩn hóa activation của mỗi layer về mean=0, std=1 trong từng mini-batch, sau đó rescale bằng hai tham số học được γ và β. Nó hiệu quả vì: (1) Giảm internal covariate shift — phân phối activation không thay đổi quá nhiều qua các vòng lặp, cho phép dùng learning rate lớn hơn mà không bị diverge; (2) Giảm nhạy cảm với weight initialization; (3) Có tác dụng regularization nhẹ vì normalization theo batch tạo noise; (4) Giúp gradient lan truyền tốt hơn qua nhiều layers."

---

### Q8: Giải thích trade-off giữa precision và recall. Trong bài toán Cat vs. Dog, cái nào quan trọng hơn?

**Trả lời:**
> "Precision = trong số ảnh model dự đoán là Dog, bao nhiêu thực sự là Dog. Recall = trong số ảnh thực sự là Dog, model tìm được bao nhiêu. Khi tăng threshold phân loại, precision tăng nhưng recall giảm, và ngược lại. Trong bài toán Cat vs. Dog thuần túy, không có hậu quả nghiêm trọng khi sai theo chiều nào — cả hai đều quan trọng như nhau, nên dùng F1 macro làm metric chính. Nhưng nếu đây là ứng dụng thực tế (ví dụ lọc ảnh chó cho trại giữ chó), FN (bỏ sót chó) có thể tệ hơn FP → ưu tiên Recall."

---

### Q9: Random Forest hoạt động như thế nào? Tại sao nó tốt hơn một Decision Tree đơn?

**Trả lời:**
> "Random Forest là ensemble của nhiều Decision Tree, mỗi tree được train trên một bootstrap sample (lấy mẫu có thay thế) và tại mỗi split chỉ xét sqrt(d) features ngẫu nhiên. Kết quả cuối là majority vote (classification) hoặc average (regression). Nó tốt hơn single tree vì: Decision Tree đơn có variance rất cao — thay đổi nhỏ trong data dẫn đến tree hoàn toàn khác. Bagging giảm variance bằng cách average nhiều trees. Random feature subsampling đảm bảo các trees không tương quan với nhau — nếu các trees giống nhau, average không giúp ích gì."

---

### Q10: Tại sao bài toán dùng Cross-Entropy Loss thay vì MSE cho classification?

**Trả lời:**
> "MSE cho classification có hai vấn đề: (1) Gradient của MSE gần bằng 0 khi prediction xa 0 hoặc xa 1 nhưng sai (ví dụ predict 0.01 cho nhãn 1) — gradient saturation làm learning rất chậm. (2) MSE không phù hợp về mặt probabilistic — nó không tương ứng với maximum likelihood estimation cho distribution Bernoulli. Cross-Entropy `L = -[y·log(p) + (1-y)·log(1-p)]` không bị gradient saturation vì `∂L/∂p = (p-y)/(p(1-p))` và khi kết hợp với sigmoid thì gradient = `(p-y)` — đơn giản và không bị triệt tiêu."

---

### Q11: Nếu dataset imbalanced (90% cat, 10% dog), bạn sẽ xử lý thế nào?

**Trả lời:**
> "Nhiều hướng tiếp cận: (1) Resampling — oversampling class thiểu số (SMOTE) hoặc undersampling class đa số; (2) Class weights — trong Logistic Regression dùng `class_weight='balanced'`, trong CrossEntropyLoss truyền `weight` tensor; (3) Thay đổi metric đánh giá từ Accuracy sang F1 macro hoặc Balanced Accuracy; (4) Threshold adjustment — thay vì ngưỡng 0.5, chọn ngưỡng tối ưu trên validation set. Trong dự án này, dataset gần như perfectly balanced (~50/50) nên không cần xử lý gì thêm."

---

### Q12: Global Average Pooling là gì? Tại sao dùng nó thay vì Flatten?

**Trả lời:**
> "Sau convolutional layers, ta có feature map kích thước (C×H×W). Flatten sẽ cho ra vector C×H×W chiều — rất lớn và nhiều tham số. Global Average Pooling tính average của mỗi feature map: output là vector C chiều (một số cho mỗi channel). Lợi ích: (1) Cực kỳ ít tham số — không có trọng số nào trong GAP; (2) Bất biến với kích thước ảnh đầu vào; (3) Có tác dụng regularization tự nhiên; (4) Mỗi output value có semantic rõ ràng — 'mức độ hiện diện của đặc trưng k trong toàn bộ ảnh'. EfficientNet-B0 dùng GAP → output 1280-dim, đó là feature vector dùng trong Classical Pipeline."

---

*Tài liệu được tổng hợp từ dự án Cat vs. Dog Image Classification — CO3117, HCMUT, Semester II 2025–2026.*
