# -*- coding: utf-8 -*-
"""
ICDAR 2026 - CircleID: Writer Identification
Private Leaderboard Rank: 19th
Model: EfficientNet-B3 with custom classifier head
"""

# =========================
# IMPORTS
# =========================
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# =========================
# CONFIG
# =========================
DATA_DIR                 = "./data/"
OUTPUT_DIR               = "./output"
EPOCHS                   = 10
BATCH_SIZE               = 32
LEARNING_RATE            = 1e-4
IMG_SIZE                 = 300
SEED                     = 0
VAL_FRAC                 = 0.2
WRITER_UNKNOWN_THRESHOLD = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SEED
# =========================
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(SEED)

# =========================
# LOAD DATA
# =========================
dataset_dir = Path(DATA_DIR)

train_df = pd.read_csv(dataset_dir / "train.csv")
extra_df = pd.read_csv(dataset_dir / "additional_train.csv")
test_df  = pd.read_csv(dataset_dir / "test.csv")

train_df = pd.concat([train_df, extra_df]).reset_index(drop=True)

# =========================
# LABEL MAP
# =========================
labels    = sorted(train_df["writer_id"].astype(str).unique())
label_map = {l: i for i, l in enumerate(labels)}
idx_map   = {i: l for l, i in label_map.items()}

train_df["y"] = train_df["writer_id"].astype(str).map(label_map).astype(int)

# =========================
# STRATIFIED SPLIT
# =========================
train_df, val_df = train_test_split(
    train_df,
    test_size=VAL_FRAC,
    stratify=train_df["y"],
    random_state=SEED
)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# DATASET
# =========================
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

class CircleDataset(Dataset):
    def __init__(self, df, root, train=True):
        self.df    = df.reset_index(drop=True)
        self.root  = Path(root)
        self.train = train

        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row      = self.df.iloc[i]
        img_path = self.root / row["image_path"]
        img      = Image.open(img_path).convert("RGB")
        x        = self.transforms(img)

        if "y" in row:
            return x, int(row["y"])
        else:
            return x, row["image_id"]

# =========================
# DATALOADERS
# =========================
train_ds = CircleDataset(train_df, dataset_dir, train=True)
val_ds   = CircleDataset(val_df,   dataset_dir, train=False)
test_ds  = CircleDataset(test_df,  dataset_dir, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# =========================
# MODEL
# =========================
def build_model(num_classes: int) -> nn.Module:
    model       = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.BatchNorm1d(in_features),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

model     = build_model(len(label_map)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# TRAIN / EVAL FUNCTIONS
# =========================
def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        preds  = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

    return correct / total

# =========================
# TRAIN LOOP
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader)
    val_acc    = evaluate(model, val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
        print(f"  → Saved best model (val_acc={val_acc:.4f})")

print(f"\nBest Val Accuracy: {best_acc:.4f}")

# =========================
# INFERENCE
# =========================
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best.pt")))

@torch.no_grad()
def predict(model, loader):
    model.eval()
    outputs = []

    for x, ids in tqdm(loader, desc="Predicting"):
        x      = x.to(device)
        probs  = F.softmax(model(x), dim=1)
        confs, preds = probs.max(dim=1)

        for img_id, conf, pred in zip(ids, confs.cpu().numpy(), preds.cpu().numpy()):
            if conf < WRITER_UNKNOWN_THRESHOLD:
                outputs.append((img_id, "-1"))
            else:
                outputs.append((img_id, idx_map[int(pred)]))

    return outputs

# =========================
# SUBMISSION
# =========================
preds = predict(model, test_loader)
sub   = pd.DataFrame(preds, columns=["image_id", "writer_id"])
sub.to_csv(os.path.join(OUTPUT_DIR, "submission_writer.csv"), index=False)

print(f"Submission saved to {OUTPUT_DIR}/submission_writer.csv 🚀")
