import os
import json
from ultralytics import YOLO

TEST_TXT = "data/processed/bstld_oneclass/test.txt"
WEIGHTS = "runs/detect/results/train/yolov8n_bstld_640/weights/best.pt"
OUT_JSON = "results/trained_eval_640/sweep_metrics.json"

CONF_LIST = [0.15, 0.25, 0.40]
IOU_LIST = [0.45, 0.60]

def read_gt_yolo(label_path):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            boxes.append([x1, y1, x2, y2])
    return boxes

def xyxy_norm(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union

with open(TEST_TXT, "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

model = YOLO(WEIGHTS)
all_results = []

for conf in CONF_LIST:
    for iou_match in IOU_LIST:
        tp = 0
        fp = 0
        fn = 0

        for image_path in image_paths:
            label_path = os.path.splitext(image_path)[0] + ".txt"
            gt_boxes = read_gt_yolo(label_path)

            results = model.predict(
                source=image_path,
                conf=conf,
                imgsz=640,
                verbose=False,
                save=False
            )

            r = results[0]
            img_h, img_w = r.orig_shape
            pred_boxes = []

            if r.boxes is not None:
                cls_list = r.boxes.cls.tolist()
                xyxy_list = r.boxes.xyxy.tolist()
                for c, b in zip(cls_list, xyxy_list):
                    if int(c) != 0:
                        continue
                    pred_boxes.append(xyxy_norm(b, img_w, img_h))

            matched_gt = set()
            matched_pred = set()

            for pi in range(len(pred_boxes)):
                best_iou = 0.0
                best_gi = -1
                for gi in range(len(gt_boxes)):
                    if gi in matched_gt:
                        continue
                    cur_iou = iou(pred_boxes[pi], gt_boxes[gi])
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_gi = gi
                if best_gi != -1 and best_iou >= iou_match:
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)

            tp += len(matched_pred)
            fp += len(pred_boxes) - len(matched_pred)
            fn += len(gt_boxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        row = {
            "conf": conf,
            "iou_match": iou_match,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        all_results.append(row)
        print(row)

os.makedirs("results/trained_eval_640", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
