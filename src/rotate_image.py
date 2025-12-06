import os
import argparse
from glob import glob
import cv2
import math
import random
import csv

EPS_MIN_AREA = 4

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Папка с исходными изображениями")
    p.add_argument("--labels_dir", required=True, help="Папка с YOLO .txt метками")
    p.add_argument("--out_dir", required=True, help="Папка для результирующего датасета")
    p.add_argument("--max_angle", type=float, default=90.0, help="Максимальный угол для случайных поворотов (abs).")
    p.add_argument("--seed", type=int, default=1234, help="Seed для воспроизводимости")
    p.add_argument("--expand", action="store_true", help="Если указано — расширять холст чтобы не обрезать изображение")
    p.add_argument("--image_ext", default=".jpg", help="Расширение выходных изображений (по умолчанию .jpg)")
    p.add_argument("--min_area", type=float, default=EPS_MIN_AREA, help="Минимальная площадь bbox (px) после поворота")
    return p.parse_args()

def list_images(images_dir):
    files = sorted(glob(os.path.join(images_dir, "*")))
    imgs = [p for p in files if os.path.splitext(p)[1].lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]]
    return imgs

def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = parts[0]
            x_c = float(parts[1]) * img_w
            y_c = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = x_c - w/2
            y1 = y_c - h/2
            x2 = x_c + w/2
            y2 = y_c + h/2
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def yolo_boxes_from_xyxy(cls, x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    x_c = x1 + w/2
    y_c = y1 + h/2
    return f"{cls} {x_c/img_w:.6f} {y_c/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"

def rotate_points(points, M):
    res = []
    for (x,y) in points:
        xp = M[0,0]*x + M[0,1]*y + M[0,2]
        yp = M[1,0]*x + M[1,1]*y + M[1,2]
        res.append((xp, yp))
    return res

def rotate_image_and_labels(img, boxes, angle, expand=False, borderMode=cv2.BORDER_CONSTANT, min_area=EPS_MIN_AREA):
    (h, w) = img.shape[:2]
    (cx, cy) = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    if expand:
        corners = [(0,0), (w-1,0), (w-1,h-1), (0,h-1)]
        rot_c = rotate_points(corners, M)
        xs = [p[0] for p in rot_c]
        ys = [p[1] for p in rot_c]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        new_w = int(math.ceil(max_x - min_x + 1))
        new_h = int(math.ceil(max_y - min_y + 1))
        tx = -min_x
        ty = -min_y
        M[0,2] += tx
        M[1,2] += ty
        out_w, out_h = new_w, new_h
    else:
        out_w, out_h = w, h

    rotated = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=borderMode)

    new_boxes = []
    for (cls, x1, y1, x2, y2) in boxes:
        pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        rpts = rotate_points(pts, M)
        xs = [p[0] for p in rpts]
        ys = [p[1] for p in rpts]
        nx1, ny1 = min(xs), min(ys)
        nx2, ny2 = max(xs), max(ys)
        nx1_cl = max(0.0, min(nx1, out_w-1))
        ny1_cl = max(0.0, min(ny1, out_h-1))
        nx2_cl = max(0.0, min(nx2, out_w-1))
        ny2_cl = max(0.0, min(ny2, out_h-1))
        area = max(0.0, nx2_cl - nx1_cl) * max(0.0, ny2_cl - ny1_cl)
        new_boxes.append((cls, nx1_cl, ny1_cl, nx2_cl, ny2_cl, area))

    # фильтруем по площади и преобразуем в YOLO-строки
    out_lines = []
    for (cls, nx1, ny1, nx2, ny2, area) in new_boxes:
        if area < min_area:
            continue
        line = yolo_boxes_from_xyxy(cls, nx1, ny1, nx2, ny2, out_w, out_h)
        if line:
            out_lines.append(line)

    return rotated, out_lines

def make_distribution(N, seed=1234):
    """
    Возвращает три множества индексов: idx_180, idx_random_angle, idx_0
    idx_random_angle содержит индексы выбранные для случайных углов.
    """
    if N == 0:
        return set(), set(), set()
    cnt_180 = int(math.floor(0.25 * N))
    cnt_random = int(math.floor(0.25 * N))
    cnt_0 = N - cnt_180 - cnt_random
    idxs = list(range(N))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    idx_180 = set(idxs[:cnt_180])
    idx_random = set(idxs[cnt_180:cnt_180+cnt_random])
    idx_0 = set(idxs[cnt_180+cnt_random:])
    return idx_180, idx_random, idx_0

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def process(args):
    random.seed(args.seed)
    imgs = list_images(args.images_dir)
    if len(imgs) == 0:
        print("Не найдено изображений в", args.images_dir)
        return

    N = len(imgs)
    idx_180, idx_random, idx_0 = make_distribution(N, seed=args.seed)
    # split idx_random into two halves for + и - знаки (воспроизводимо)
    idx_random_list = sorted(list(idx_random))
    half = len(idx_random_list) // 2
    idx_random_pos = set(idx_random_list[:half])
    idx_random_neg = set(idx_random_list[half:])

    out_images = os.path.join(args.out_dir, "images")
    out_labels = os.path.join(args.out_dir, "labels")
    ensure_dir(out_images)
    ensure_dir(out_labels)

    mapping = []

    for i, img_path in enumerate(imgs):
        fname = os.path.basename(img_path)
        name_noext = os.path.splitext(fname)[0]
        lbl_path = os.path.join(args.labels_dir, name_noext + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            print("Warning: can't read", img_path)
            continue
        h,w = img.shape[:2]
        boxes = load_yolo_labels(lbl_path, w, h)

        if i in idx_180:
            angle = 180.0
        elif i in idx_random_pos:
            angle = random.uniform(0.0, args.max_angle)
        elif i in idx_random_neg:
            angle = -random.uniform(0.0, args.max_angle)
        else:
            angle = 0.0

        if angle == 0.0:
            # копирование без изменения (унифицируем расширение)
            out_img_name = name_noext + args.image_ext if os.path.splitext(fname)[1].lower() != args.image_ext.lower() else fname
            out_img_path = os.path.join(out_images, out_img_name)
            cv2.imwrite(out_img_path, img)
            out_lbl_path = os.path.join(out_labels, os.path.splitext(out_img_name)[0] + ".txt")
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as fin, open(out_lbl_path, "w", encoding="utf-8") as fout:
                    fout.write(fin.read())
            else:
                open(out_lbl_path, "w", encoding="utf-8").close()
            mapping.append((os.path.basename(img_path), os.path.basename(out_img_path), angle))
        else:
            rotated, out_lines = rotate_image_and_labels(img, boxes, angle, expand=args.expand, min_area=args.min_area)
            angle_tag = f"rot{int(round(angle))}" if abs(angle - round(angle)) < 1e-6 else f"rot{angle:.2f}"
            # имя включает знак и угол
            sign = "p" if angle > 0 else "n"
            out_img_name = f"{name_noext}__{sign}{angle_tag}{args.image_ext}"
            out_img_path = os.path.join(out_images, out_img_name)
            cv2.imwrite(out_img_path, rotated)
            out_lbl_path = os.path.join(out_labels, os.path.splitext(out_img_name)[0] + ".txt")
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                if len(out_lines) > 0:
                    f.write("\n".join(out_lines))
            mapping.append((os.path.basename(img_path), os.path.basename(out_img_path), angle))

    # save mapping csv
    map_csv = os.path.join(args.out_dir, "mapping.csv")
    with open(map_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original", "output", "angle"])
        for r in mapping:
            writer.writerow(r)

    # summary
    cnts = {"rot180":0, "rand_pos":0, "rand_neg":0, "rot0":0}
    for _, out_name, angle in mapping:
        if angle == 180.0:
            cnts["rot180"] += 1
        elif angle == 0.0:
            cnts["rot0"] += 1
        elif angle > 0:
            cnts["rand_pos"] += 1
        else:
            cnts["rand_neg"] += 1

    print("Done. Summary:")
    print(f" Total images: {len(mapping)}")
    print(f" 180 deg: {cnts['rot180']}")
    print(f" random +: {cnts['rand_pos']}")
    print(f" random -: {cnts['rand_neg']}")
    print(f" 0 deg (unchanged): {cnts['rot0']}")
    print(f" Mapping CSV: {map_csv}")

if __name__ == "__main__":
    args = parse_args()
    if args.max_angle < 0 or args.max_angle > 180:
        raise SystemExit("max_angle должен быть в диапазоне [0, 180]")
    process(args)