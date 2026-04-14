# ICDAR 2026 - CircleID: Writer Identification
### 🏆 Private Leaderboard Rank: 19th

Solution for the [ICDAR 2026 CircleID Writer Identification](https://kaggle.com/competitions/icdar-2026-circleid-writer-identification) competition on Kaggle.

> **Task:** Identify who drew a circle using only a scanned image of a hand-drawn circle. The test set includes both known writers (seen during training) and unknown writers (must be predicted as `-1`).

---

## 📋 Approach

### Model
- **Backbone:** EfficientNet-B3 (ImageNet pretrained via `torchvision`)
- **Custom Classifier Head:**
  ```
  Linear(in_features → in_features)
  BatchNorm1d
  ReLU
  Dropout(0.4)
  Linear(in_features → num_classes)
  ```

### Training Details
| Parameter | Value |
|---|---|
| Image Size | 300 × 300 |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Val Split | 20% stratified |

### Data
- Combined `train.csv` + `additional_train.csv` for training
- Augmentations: `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation(20)`, `ColorJitter`, `RandomGrayscale`

### Unknown Writer Detection
Writers not seen during training are predicted as `-1` using a **confidence threshold of 0.6** on the softmax output. If the model's top prediction confidence is below this threshold, the writer is classified as unknown.

---

## 🗂️ Repository Structure

```
circleid-writer-identification/
├── train.py            # Main training + inference script
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download competition data
Download the dataset from the [Kaggle competition page](https://kaggle.com/competitions/icdar-2026-circleid-writer-identification) and place it in a `data/` folder:
```
data/
├── train.csv
├── additional_train.csv
├── test.csv
└── <image folders>
```

### 3. Train and generate submission
```bash
python train.py
```

Output will be saved to `output/submission_writer.csv`.

---

## 📊 Results

| Split | Score |
|---|---|
| Public Leaderboard | — |
| **Private Leaderboard** | **Top 19** |

---

## 🔧 Environment
- Python 3.10+
- PyTorch 2.x
- Google Colab (A100 / T4 GPU)

---

## 📄 Citation
```bibtex
@misc{circleid2026,
  title  = {ICDAR 2026 - CircleID: Writer Identification},
  author = {Thomas Gorges},
  year   = {2026},
  url    = {https://kaggle.com/competitions/icdar-2026-circleid-writer-identification}
}
```
