#!/usr/bin/env python3
"""
merge_yolo_dirs.py

Переносит/копирует пары (image, label) из src -> dst, безопасно обрабатывая конфликты имён.

Примеры:
# dry-run, посмотреть что произойдет
python merge_yolo_dirs.py --src-images /path/src/images --src-labels /path/src/labels \
    --dst-images /path/dst/images --dst-labels /path/dst/labels --dry-run

# выполнить перемещение (move)
python merge_yolo_dirs.py --src-images /path/src/images --src-labels /path/src/labels \
    --dst-images /path/dst/images --dst-labels /path/dst/labels

# копировать вместо перемещения
python merge_yolo_dirs.py ... --mode copy

# перезаписывать файлы в dst (внимание) — иначе используются суффиксы
python merge_yolo_dirs.py ... --overwrite

"""
from pathlib import Path
import argparse
import shutil
import itertools
import datetime
import sys

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(path: Path):
    return path.suffix.lower() in IMG_EXTS

def find_unique_path(dst_dir: Path, stem: str, suffix: str, overwrite: bool):
    """
    Возвращает (target_path, renamed) где renamed == True если имя изменилось.
    """
    target = dst_dir / (stem + suffix)
    if not target.exists() or overwrite:
        return target, False
    # иначе ищем _dupNNNN
    for i in itertools.count(1):
        target = dst_dir / f"{stem}_dup{i:04d}{suffix}"
        if not target.exists():
            return target, True

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser(description="Merge YOLO image/label folders safely.")
    p.add_argument("--src-images", required=True, help="source images directory")
    p.add_argument("--src-labels", required=True, help="source labels directory")
    p.add_argument("--dst-images", required=True, help="destination images directory")
    p.add_argument("--dst-labels", required=True, help="destination labels directory")
    p.add_argument("--mode", choices=("move","copy"), default="copy", help="move (default) or copy")
    p.add_argument("--overwrite", action="store_true", help="overwrite files in destination (if set). Otherwise generate unique names")
    p.add_argument("--dry-run", action="store_true", help="show actions but don't perform")
    p.add_argument("--require-label", action="store_true", help="skip images that don't have a label in src_labels")
    p.add_argument("--move-unpaired-labels", action="store_true", help="also move labels that don't have matching images (default: yes). If not set, they will be left in source.")
    p.add_argument("--backup", action="store_true", help="when using --overwrite, create backup of overwritten files in dst before overwrite")
    return p.parse_args()

def backup_file(path: Path, backup_root: Path):
    ensure_dir(backup_root)
    tgt = backup_root / path.name
    shutil.copy2(path, tgt)

def main():
    args = parse_args()
    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    dst_images = Path(args.dst_images)
    dst_labels = Path(args.dst_labels)

    for p in (src_images, src_labels):
        if not p.exists():
            print(f"ERROR: {p} does not exist.", file=sys.stderr)
            sys.exit(1)

    ensure_dir(dst_images)
    ensure_dir(dst_labels)

    # prepare backup folder if needed
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = None
    if args.backup and args.overwrite:
        backup_root = dst_images.parent / f"dst_backup_{timestamp}"
        ensure_dir(backup_root)
        print(f"Backup root for overwritten files: {backup_root}")

    # gather source images
    src_imgs = [p for p in src_images.iterdir() if p.is_file() and is_image(p)]
    src_imgs.sort()
    total = len(src_imgs)
    moved = 0
    skipped_no_label = 0
    renamed_count = 0

    print(f"Found {total} images in source {src_images}")

    for img_path in src_imgs:
        stem = img_path.stem
        label_path = src_labels / (stem + ".txt")
        if args.require_label and not label_path.exists():
            print(f"SKIP (no label): {img_path.name}")
            skipped_no_label += 1
            continue

        # determine destination image path
        target_img, img_renamed = find_unique_path(dst_images, stem, img_path.suffix.lower(), args.overwrite)
        # determine destination label path (matching stem)
        target_lbl, lbl_renamed = find_unique_path(dst_labels, target_img.stem, ".txt", args.overwrite)

        # if overwriting and backup requested, backup existing files
        if args.overwrite and args.backup:
            if target_img.exists():
                backup_file(target_img, backup_root)
            if target_lbl.exists():
                backup_file(target_lbl, backup_root)

        action_str = f"{'COPY' if args.mode=='copy' else 'MOVE'} image {img_path} -> {target_img}"
        if label_path.exists():
            action_str += f", label {label_path} -> {target_lbl}"
        else:
            action_str += " (no label found)"

        if args.dry_run:
            print("[DRY] " + action_str + ( " [RENAME]" if img_renamed or lbl_renamed else "" ))
        else:
            # perform action
            ensure_dir(target_img.parent)
            ensure_dir(target_lbl.parent)
            if args.mode == "move":
                shutil.move(str(img_path), str(target_img))
            else:
                shutil.copy2(str(img_path), str(target_img))
            if label_path.exists():
                if args.mode == "move":
                    shutil.move(str(label_path), str(target_lbl))
                else:
                    shutil.copy2(str(label_path), str(target_lbl))
            if img_renamed or lbl_renamed:
                renamed_count += 1
            moved += 1
            print(action_str + ( " [RENAMED]" if img_renamed or lbl_renamed else "" ))

    # handle unpaired labels if requested
    if args.move_unpaired_labels:
        src_lbls = [p for p in src_labels.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
        # exclude those which we already moved (they no longer exist if moved), so list remaining
        remaining = [p for p in src_lbls if (src_images / (p.stem + ".jpg")).exists() == False and (src_images / (p.stem + ".png")).exists() == False]
        for lbl in remaining:
            target_lbl, lbl_renamed = find_unique_path(dst_labels, lbl.stem, ".txt", args.overwrite)
            if args.dry_run:
                print(f"[DRY] {'COPY' if args.mode=='copy' else 'MOVE'} label {lbl} -> {target_lbl}")
            else:
                if args.mode == "move":
                    shutil.move(str(lbl), str(target_lbl))
                else:
                    shutil.copy2(str(lbl), str(target_lbl))
                print(f"Moved leftover label {lbl.name} -> {target_lbl.name}" + (" [RENAMED]" if lbl_renamed else ""))

    print("Done.")
    print(f"Images processed: {total}, moved/copied: {moved}, renamed pairs: {renamed_count}, skipped (no label): {skipped_no_label}")

if __name__ == "__main__":
    main()
