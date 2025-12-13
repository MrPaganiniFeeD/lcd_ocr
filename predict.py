import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageOps
from ultralytics import YOLO
from model import ResNetPredictor
import numpy as np
import sys

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def get_image_paths(indir: Path) -> List[Path]:
    return [p for p in indir.rglob('*') if p.suffix.lower() in VALID_EXTS and p.is_file()]


def parse_args():
    p = argparse.ArgumentParser(description="Batch detect + save annotated images and results")
    p.add_argument('--indir', required=True, help='Папка с изображениями (будут обработаны рекурсивно)')
    p.add_argument('--outdir', required=True, help='Папка для сохранения разметки и результатов')
    p.add_argument('--yolo', default=r"weight\super_big\epoch20.pt", help='Путь до .pt модели YOLO (Ultralytics)')
    p.add_argument('--class_resnet', dest='resnet_class_model', default=r"weight\best_lcd_resnet183.pth", help='Путь до модели классификатора (ResNetPredictor)')
    p.add_argument('--class_yolo', dest='yolo_class_model', default=r"weight\class_yolo_epoch25.pt", help='Путь до модели классификатора (YOLO)')
    p.add_argument('--class_is_yolo', required=True, help='True or False')
    p.add_argument('--iou', type=float, default=0.1, help='IOU для предсказаний YOLO (default 0.3)')
    p.add_argument('--conf_inverted', type=float, default=0.9, help='0.0-1.0')    

    return p.parse_args()


def rotate_image(image, class_pred):
    image = ImageOps.exif_transpose(image)
    if class_pred == 1:
        image = image.rotate(-180)
    return image


def detect_temperature_single(image_path: Path, yolo_detect_model: YOLO, class_model: ResNetPredictor, yolo_class_model, yolo_class_flag: bool ,iou: float=0.3, conf_inverted: int=0.5):
    """
    - применяет классификатор для поворота
    - запускает детекцию
    - возвращает (temperature_or_None, detections_list, annotated_image_array)
    detections_list: list of dicts {'class_id': int, 'class_name': str, 'conf': float, 'bbox': [x1,y1,x2,y2]}
    """
    try:
        if yolo_class_flag == False:
            class_pred = class_model.predict(str(image_path))["pred"]
        else:
            results = yolo_class_model(str(image_path), verbose=False)
            class_pred = results[0].probs.top1  # 0 или 1
            conf = results[0].probs.data
            print(conf)          
            if class_pred == 0 and conf[0] >= conf_inverted:
                class_pred = 1
            else:
                class_pred = 0
    except Exception as e:
        print(f"[WARN] Ошибка классификатора для {image_path}: {e}")
        class_pred = None

    image = Image.open(image_path)
    image = rotate_image(image, class_pred)

    results = yolo_detect_model.predict(image, iou=iou)
    if len(results) == 0:
        return None, [], np.array(image)

    result = results[0]

    out_detection = []
    for i, box in enumerate(result.boxes):
        try:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            cls_name = result.names[class_id]
            bbox = [float(x) for x in box.xyxy[0].tolist()]  # [x1,y1,x2,y2]
        except Exception:
            continue
        out_detection.append({
            'class_id': class_id,
            'class_name': cls_name,
            'conf': confidence,
            'bbox': bbox
        })

    centers_and_names = []
    for d in out_detection:
        x1, y1, x2, y2 = d['bbox']
        centers_and_names.append(((x1 + x2) / 2, d['class_name']))

    centers_and_names.sort(key=lambda x: x[0])
    out_value_str = ''
    is_node = False
    for center, name in centers_and_names:
        if name == 'dot' and is_node:
            continue
        if name == 'c':
            continue
        if name == 'dot' and not is_node:
            is_node = True
            out_value_str += "."
            continue
        out_value_str += name
        

    out_value = None
    try:
        if out_value_str != '':
            out_value = float(out_value_str)

    except ValueError:
        out_value = None

    try:
        annotated = result.plot()  # numpy array (H,W,3)
    except Exception:
        annotated = np.array(image)

    if (out_value != None) and (out_value > 30):
        out_value = None

    return out_value, out_detection, annotated


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    yolo_path = args.yolo
    class_resnet_path = args.resnet_class_model
    yolo_class_path = args.yolo_class_model
    class_is_yolo = args.class_is_yolo
    conf_inverted = args.conf_inverted

    if not indir.exists():
        print("indir не существует:", indir)
        sys.exit(1)

    ensure_dir(outdir)
    images = get_image_paths(indir)
    if len(images) == 0:
        print("Нет изображений в указанной папке.")
        sys.exit(0)

    print("Загружаем модели...")
    yolo_model = YOLO(yolo_path)
    class_model = ResNetPredictor(class_resnet_path)
    yolo_class_model = YOLO(yolo_class_path)

    csv_path = outdir / 'results.csv'
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'temperature', 'error', 'num_detections'])

    print(f"Обработка {len(images)} файлов...")
    for img_path in images:
        rel = img_path.relative_to(indir)
        out_img_path = outdir / rel.name 
        out_json_path = outdir / (rel.stem + '.json')

        print(f"-> {img_path} ...", end=' ')
        try:
            temp, detections, annotated_np = detect_temperature_single(img_path, yolo_model, class_model, yolo_class_model, class_is_yolo, iou=args.iou, conf_inverted=conf_inverted)
            try:
                annotated_img = Image.fromarray(annotated_np)
                annotated_img.save(out_img_path)
            except Exception as e:
                print(f"[WARN] не удалось сохранить изображение: {e}")

            json_data = {
                'input': str(img_path),
                'temperature': temp,
                'num_detections': len(detections),
                'detections': detections
            }
            try:
                with open(out_json_path, 'w', encoding='utf-8') as jf:
                    json.dump(json_data, jf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] не удалось сохранить json: {e}")

            error_flag = (temp is None)
            csv_writer.writerow([str(img_path.name), temp if temp is not None else '', error_flag, len(detections)])

            print(f"сделано. temp={temp}, dets={len(detections)}")
        except Exception as e:
            print(f"Ошибка при обработке: {e}")
            csv_writer.writerow([str(img_path.name), '', True, 0])

    csv_file.close()
    print("Готово. Результаты в:", outdir)


if __name__ == '__main__':
    main()
