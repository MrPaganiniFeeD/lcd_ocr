# simple_test.py
import os
import argparse
import csv
import time
from pathlib import Path


import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

def build_resnet18_single_logit():
    net = models.resnet18(pretrained=False)
    in_feats = net.fc.in_features
    net.fc = nn.Linear(in_feats, 1)  # один логит для бинарной классификации
    return net

def load_checkpoint_flex(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    # возможные форматы чекпоинта
    if isinstance(ck, dict):
        # принять разные ключи
        if 'model_state_dict' in ck:
            state = ck['model_state_dict']
        elif 'model_state' in ck:
            state = ck['model_state']
        elif 'state_dict' in ck:
            state = ck['state_dict']
        else:
            # возможно это прямо state_dict (словарь с тензорами)
            # проверим — есть ли там ключи типа 'fc.weight' или 'layer1.0.conv1.weight'
            sample_keys = list(ck.keys())
            if any(k.startswith('layer') or k.startswith('fc') or 'conv' in k for k in sample_keys):
                state = ck
            else:
                raise RuntimeError(f"Не удалось определить веса в чекпойнте. Доступные ключи: {list(ck.keys())[:10]}")
    else:
        raise RuntimeError("Чекпойнт в неожиданном формате")

    # убрать возможный префикс 'module.' из ключей (если сохраняли через DataParallel)
    new_state = {}
    for k, v in state.items():
        nk = k
        if k.startswith('module.'):
            nk = k[len('module.'):]
        new_state[nk] = v

    # попытаемся загрузить state_dict; если несовпадение размеров — выдадим читаемую ошибку
    try:
        model.load_state_dict(new_state)
    except RuntimeError as e:
        # пробуем загрузить с strict=False, чтобы увидеть несовпадения
        missing_keys = None
        unexpected_keys = None
        try:
            missing, unexpected = model.load_state_dict(new_state, strict=False)
        except Exception:
            missing, unexpected = None, None
        raise RuntimeError(f"Ошибка загрузки state_dict: {e}\n(Попытка загрузки с strict=False вернула missing={missing}, unexpected={unexpected})")
    return model

def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # допустимые имена столбцов для filename и label
        fieldnames = [n.strip() for n in reader.fieldnames]
        # определим имена колонок
        filename_col = None
        label_col = None
        for c in fieldnames:
            low = c.lower()
            if low in ('filename','file','name','path'):
                filename_col = c
            if low in ('flip','label','orientation'):
                label_col = c
        if filename_col is None:
            raise ValueError(f"CSV не содержит столбца filename. Колонки: {fieldnames}")
        # теперь читаем
        f.seek(0)
        reader = csv.DictReader(f)
        for r in reader:
            fn = r[filename_col].strip()
            lbl = None
            if label_col and r.get(label_col) is not None and r[label_col] != '':
                try:
                    lbl = int(r[label_col])
                except:
                    lbl = None
            rows.append((fn, lbl))
    return rows

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # трансформация
    tf = get_transform(img_size=args.img_size)

    # модель
    model = build_resnet18_single_logit()
    model = load_checkpoint_flex(model, args.model, device)
    model.to(device).eval()

    # читаем CSV
    rows = read_csv_rows(args.csv)
    if len(rows) == 0:
        print("CSV пустой или не прочитан.")
        return

    n_total = 0
    n_missing = 0
    tp = fp = tn = fnt = 0
    out_rows = []  # сохранить результаты

    start_time = time.time()
    for fn, lbl in rows:
        # формируем путь к файлу
        img_path = os.path.join(args.images_dir, fn)
        if not os.path.exists(img_path):
            # попробуем basename
            base = os.path.basename(fn)
            img_path2 = os.path.join(args.images_dir, base)
            if os.path.exists(img_path2):
                img_path = img_path2
            else:
                print(f"[WARN] Файл не найден: {fn} (пропускаю)")
                n_missing += 1
                continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Не удалось открыть {img_path}: {e}")
            n_missing += 1
            continue
        
        x = tf(img).unsqueeze(0).to(device)  # (1,C,H,W)
        with torch.no_grad():
            logits = model(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > args.threshold else 0

        out_rows.append((fn, prob, pred, lbl))
        n_total += 1

        # если есть метка — считаем метрики
        if lbl is not None:
            if lbl == 1 and pred == 1:
                tp += 1
            elif lbl == 0 and pred == 1:
                fp += 1
            elif lbl == 0 and pred == 0:
                tn += 1
            elif lbl == 1 and pred == 0:
                fnt += 1

    # вывод результатов
    print("inference:", time.time() - start_time)
    print("Processed:", n_total, "missing/skipped:", n_missing)
    if n_total == 0:
        print("Нет обработанных изображений.")
        return

    # если хотя бы одна метка присутствует — считаем accuracy и др.
    n_labeled = sum(1 for r in out_rows if r[3] is not None)
    if n_labeled > 0:
        acc = (tp + tn) / (tp + tn + fp + fnt) if (tp+tn+fp+fnt)>0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fnt) if (tp + fnt) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print("=== Metrics on labeled images ===")
        print(f"Total labeled: {n_labeled}")
        print(f"TP={tp}  FP={fp}  TN={tn}  FN={fnt}")
        print(f"Accuracy={acc:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    else:
        print("В CSV нет меток — режим inference-only. Сохранены предсказания.")

    # сохранить predictions.csv
    out_csv = args.out or "predictions.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','probability','pred','label_if_any'])
        for fn, prob, pred, lbl in out_rows:
            writer.writerow([fn, f"{prob:.6f}", pred, lbl if lbl is not None else ""])
    print("Saved predictions ->", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to model .pth")
    p.add_argument("--images_dir", required=True, help="folder with images")
    p.add_argument("--csv", required=True, help="csv with filename and optionally flip/label")
    p.add_argument("--out", default="predictions.csv", help="output csv path")
    p.add_argument("--img_size", type=int, default=224, help="image size for center-crop")
    p.add_argument("--threshold", type=float, default=0.5, help="probability threshold for class=1")
    args = p.parse_args()
    main(args)
