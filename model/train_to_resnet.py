# train_lcd.py
import os
import csv
from pathlib import Path
from collections import Counter
import argparse
import json
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


# ----------------------------
# Конфигурация ( можно менять / параметризовать )
# ----------------------------
DEFAULTS = {
    "dataset_root": r"C:\Users\Egor\VsCode project\lcd_display\data\new_big_aug_rotate_180",
    "train_csv": "train.csv",
    "val_csv": "val.csv",
    "test_csv": "test.csv",
    "images_subdir": "images",       # у тебя в каждом сплите папка image/
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 4,
    "model_save": r"C:\Users\Egor\VsCode project\lcd_display\weight",
    "seed": 42,
}

# ----------------------------
# Утилиты
# ----------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    # нормализуем имена колонок
    df.columns = [c.strip() for c in df.columns]
    return df

def validate_csv_images(df, images_dir, filename_col_candidates=('filename','file','name')):
    # возвращает список отсутствующих и лишних файлов
    filename_col = None
    for c in filename_col_candidates:
        if c in df.columns:
            filename_col = c
            break
    if filename_col is None:
        raise ValueError(f"CSV не содержит столбца filename. Найдены столбцы: {df.columns.tolist()}")
    filenames = df[filename_col].astype(str).str.strip().tolist()
    files_on_disk = set(os.listdir(images_dir))
    missing = [f for f in filenames if f not in files_on_disk]
    extra = [f for f in files_on_disk if f not in filenames]
    return filename_col, missing, extra

# ----------------------------
# Dataset для твоей структуры CSV
# ----------------------------
class LCDCsvDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None, filename_col='filename', label_col_candidates=('flip','label','orientation')):
        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        self.images_dir = images_dir
        self.transform = transform

        # определить столбец с меткой (flip)
        self.label_col = None
        for c in label_col_candidates:
            if c in self.df.columns:
                self.label_col = c
                break
        if self.label_col is None:
            raise ValueError(f"CSV не содержит столбца flip/label. Найдены: {self.df.columns.tolist()}")

        if filename_col not in self.df.columns:
            raise ValueError(f"CSV не содержит столбца {filename_col}. Доступные: {self.df.columns.tolist()}")
        self.filename_col = filename_col

        # приведение типов
        self.df[self.filename_col] = self.df[self.filename_col].astype(str).str.strip()
        self.df[self.label_col] = self.df[self.label_col].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row[self.filename_col]
        path = os.path.join(self.images_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row[self.label_col])
        return img, label

# ----------------------------
# Трансформации (без 180° поворота)
# ----------------------------
def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15)], p=0.8),
        transforms.RandomAffine(degrees=8, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

# ----------------------------
# DataLoaders (Weighted sampler для train)
# ----------------------------
def make_loaders(root, train_csv, val_csv, test_csv, images_subdir='image', img_size=224, batch_size=32, num_workers=4):
    train_images = os.path.join(root, 'train', images_subdir)
    val_images = os.path.join(root, 'val', images_subdir)
    test_images = os.path.join(root, 'test', images_subdir)

    train_csv_path = os.path.join(root, 'train', train_csv)
    val_csv_path = os.path.join(root, 'val', val_csv)
    test_csv_path = os.path.join(root, 'test', test_csv)

    train_tf, val_tf = get_transforms(img_size=img_size)

    # Валидация
    train_df = read_csv(train_csv_path)
    filename_col, missing, extra = validate_csv_images(train_df, train_images)
    if missing:
        print(f"[WARNING] В train CSV {len(missing)} файлов отсутствуют на диске. Первые 10: {missing[:10]}")
    if extra:
        print(f"[INFO] В папке train найдено {len(extra)} файлов, не указанных в CSV. Первые 10: {extra[:10]}")

    # Создаём Dataset'ы
    train_ds = LCDCsvDataset(train_csv_path, train_images, transform=train_tf, filename_col=filename_col)
    val_ds = LCDCsvDataset(val_csv_path, val_images, transform=val_tf, filename_col=filename_col)
    test_ds = LCDCsvDataset(test_csv_path, test_images, transform=val_tf, filename_col=filename_col)

    # Weighted sampler для train
    labels = train_ds.df[train_ds.label_col].tolist()
    class_counts = Counter(labels)
    print("Train class counts:", class_counts)
    class_weights = {cls: 1.0/count for cls,count in class_counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.label_col

# ----------------------------
# Модель: ResNet18 -> 1 логит (BCEWithLogits)
# ----------------------------
def build_model(pretrained=True):
    net = models.resnet18(pretrained=pretrained)
    in_feats = net.fc.in_features
    net.fc = nn.Linear(in_feats, 1)  # бинарная классификация (логит)
    return net

# ----------------------------
# Тренировка
# ----------------------------
def train_and_evaluate(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    set_seed(cfg['seed'])

    train_loader, val_loader, test_loader, label_col = make_loaders(
        root=cfg['dataset_root'],
        train_csv=cfg['train_csv'],
        val_csv=cfg['val_csv'],
        test_csv=cfg['test_csv'],
        images_subdir=cfg['images_subdir'],
        img_size=cfg['img_size'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )

    model = build_model(pretrained=True).to(device)

    # Опционально: заморозить backbone на первых эпохах
    # for name, p in model.named_parameters():
    #     if "fc" not in name:
    #         p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(1, cfg['num_epochs'] + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch}/{cfg['num_epochs']}"):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            logits = model(imgs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            preds = (probs > 0.5).astype(int).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy().ravel().astype(int).tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

        # Валидация
        model.eval()
        vloss = 0.0
        vpreds = []
        vlabels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1).float()
                logits = model(imgs)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)
                loss = criterion(logits, labels)
                vloss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                vpreds.extend((probs > 0.5).astype(int).tolist())
                vlabels.extend(labels.cpu().numpy().ravel().astype(int).tolist())

        val_loss = vloss / len(val_loader.dataset)
        val_acc = accuracy_score(vlabels, vpreds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(vlabels, vpreds, average='binary', zero_division=0)
        cm = confusion_matrix(vlabels, vpreds)

        print(f"Epoch {epoch} Train: loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | Val: loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")
        print("Val Confusion Matrix:\n", cm)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            weight_name = "/best_lcd_resnet18" + str(epoch) + ".pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cfg': cfg,
                'epoch': epoch,
                'val_loss': val_loss
            }, cfg['model_save'] + weight_name)
            print(f"Saved best model to {cfg['model_save']} (val_loss={val_loss:.4f})")

    # --- Тестирование на лучшей модели ---
    print("Loading best model for final evaluation...")
    ckpt = torch.load(cfg['model_save'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    tpreds = []
    tlabels = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            logits = model(imgs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            tpreds.extend((probs > 0.5).astype(int).tolist())
            tlabels.extend(labels.cpu().numpy().ravel().astype(int).tolist())

    acc = accuracy_score(tlabels, tpreds)
    prec, rec, f1, _ = precision_recall_fscore_support(tlabels, tpreds, average='binary', zero_division=0)
    cm = confusion_matrix(tlabels, tpreds)
    report = classification_report(tlabels, tpreds, digits=4)

    print("=== Test results ===")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to json config (optional)")
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    if args.config:
        with open(args.config, 'r') as f:
            new = json.load(f)
        cfg.update(new)

    print("Configuration:")
    print(json.dumps(cfg, indent=2))
    train_and_evaluate(cfg)
