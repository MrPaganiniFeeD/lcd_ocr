#!/usr/bin/env python3
"""
relabel_yolo_move_minus_to_end.py

Перекодирует class ids YOLO (.txt) по новому маппингу и обновляет yaml names.

Использование:
 python relabel_yolo_move_minus_to_end.py --yaml path/to/data.yaml --out-dir converted_labels --update-yaml
 python relabel_yolo_move_minus_to_end.py --labels /path/to/labels --inplace --backup
"""
import argparse
import os
import shutil
from pathlib import Path
import re
import sys

# Новый маппинг: старый индекс -> новый индекс
REMAP = {
    0: 12,  # '-' -> последний индекс
    1: 11,  # '.' -> 'dot'
    2: 0,   # '0' -> 0
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9,
}

# Новые names в нужном порядке
NEW_NAMES = ['0','1','2','3','4','5','6','7','8','9','c','dot','-']

def process_file(in_path: Path, out_path: Path):
    txt = in_path.read_text(encoding='utf-8').strip()
    if not txt:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding='utf-8')
        return
    lines = txt.splitlines()
    new_lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            old_cls = int(parts[0])
        except Exception:
            print(f"Warning: can't parse class in {in_path}: '{ln}' -> skipping")
            continue
        if old_cls not in REMAP:
            print(f"Warning: old class {old_cls} in {in_path} not in remap -> skipping")
            continue
        new_cls = REMAP[old_cls]
        new_line = " ".join([str(new_cls)] + parts[1:])
        new_lines.append(new_line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding='utf-8')

def find_label_dir_from_image_dir(img_dir: Path):
    s = str(img_dir)
    if s.endswith("images"):
        return Path(s[:-len("images")] + "labels")
    parent = img_dir.parent
    guess = parent / "labels"
    return guess

def gather_label_files_from_dir(labels_dir: Path):
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.rglob("*.txt"))

def update_yaml_names(yaml_path: Path, out_yaml_path: Path = None):
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
        data['nc'] = len(NEW_NAMES)
        data['names'] = NEW_NAMES
        out_path = out_yaml_path or yaml_path
        out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
        return True
    except Exception:
        txt = yaml_path.read_text(encoding='utf-8')
        pattern = r"(?m)^names\s*:\s*\[.*?\]\s*$"
        replacement = "names: [" + ", ".join(repr(n) for n in NEW_NAMES) + "]"
        newtxt = txt
        if re.search(pattern, txt):
            newtxt = re.sub(pattern, replacement, txt)
        newtxt = re.sub(r"(?m)^nc\s*:\s*\d+\s*$", f"nc: {len(NEW_NAMES)}", newtxt)
        out_path = out_yaml_path or yaml_path
        out_path.write_text(newtxt, encoding='utf-8')
        return True

def main():
    p = argparse.ArgumentParser(description="Перекодировать YOLO метки (class ids) согласно новому порядку, где '-' в конце")
    p.add_argument("--yaml", help="путь к yaml файлу (в котором указаны пути train/val/test images)")
    p.add_argument("--labels", nargs="*", help="или явные папки с .txt метками (можно несколько)")
    p.add_argument("--out-dir", help="если указан, записывать новые метки в этот корень (поддерживает структуру)", default=None)
    p.add_argument("--inplace", action="store_true", help="перезаписать существующие .txt файлы (не рекомендуется без backup)")
    p.add_argument("--backup", action="store_true", help="сделать резервную копию labels перед изменением (если inplace)")
    p.add_argument("--update-yaml", action="store_true", help="обновить yaml names в файле (если --yaml указан)")

    args = p.parse_args()

    label_dirs = set()
    if args.yaml:
        yaml_path = Path(args.yaml)
        if not yaml_path.exists():
            print("yaml файл не найден:", yaml_path)
            sys.exit(1)
        txt = yaml_path.read_text(encoding='utf-8')
        for key in ("train","val","test"):
            m = re.search(rf"^{key}\s*:\s*(.+)$", txt, flags=re.MULTILINE)
            if m:
                path_str = m.group(1).strip().strip('\'"')
                img_dir = (yaml_path.parent / path_str) if not os.path.isabs(path_str) else Path(path_str)
                labels_guess = find_label_dir_from_image_dir(img_dir)
                label_dirs.add(str(labels_guess))
    if args.labels:
        for ld in args.labels:
            label_dirs.add(ld)

    if not label_dirs:
        print("Не найдены папки с метками. Укажите --yaml или --labels.")
        sys.exit(1)

    timestamp = None
    if args.backup and args.inplace:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for ld in sorted(label_dirs):
        labels_dir = Path(ld)
        if not labels_dir.exists():
            print(f"Warning: labels dir не существует: {labels_dir} (пропускаю)")
            continue
        files = gather_label_files_from_dir(labels_dir)
        if not files:
            print(f"Warning: нет .txt файлов в {labels_dir} (пропускаю)")
            continue

        if args.inplace:
            if args.backup:
                backup_dir = labels_dir.parent / f"{labels_dir.name}_backup_{timestamp}"
                print(f"Создаю резервную копию {labels_dir} -> {backup_dir}")
                if backup_dir.exists():
                    print("Backup dir already exists, пропускаю копирование.")
                else:
                    shutil.copytree(labels_dir, backup_dir)
            print(f"[INPLACE] Обрабатываю {len(files)} файлов в {labels_dir} ...")
            for f in files:
                process_file(f, f)
        else:
            if not args.out_dir:
                print("Нужно указать --out-dir если не используете --inplace")
                sys.exit(1)
            out_root = Path(args.out_dir)
            for f in files:
                rel = f.relative_to(labels_dir.parent)
                out_path = out_root / rel
                process_file(f, out_path)
            print(f"Записал {len(files)} файлов в {out_root}")

    if args.update_yaml and args.yaml:
        yaml_path = Path(args.yaml)
        out_yaml = yaml_path
        if args.out_dir:
            out_yaml = Path(args.out_dir) / yaml_path.name
        ok = update_yaml_names(yaml_path, out_yaml_path=out_yaml)
        if ok:
            print(f"YAML обновлён: {out_yaml}")
        else:
            print("YAML не был обновлён автоматически. Проверьте вручную.")

    print("Готово. Remap:", REMAP)

if __name__ == "__main__":
    main()
