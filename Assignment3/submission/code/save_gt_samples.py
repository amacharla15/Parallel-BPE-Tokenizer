import os
import cv2

TEST_TXT = "data/processed/bstld_oneclass/test.txt"
OUT_DIR = "results/gt_annotated_samples"
SAVE_COUNT = 10

os.makedirs(OUT_DIR, exist_ok=True)

def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = int(cx - w / 2.0)
            y1 = int(cy - h / 2.0)
            x2 = int(cx + w / 2.0)
            y2 = int(cy + h / 2.0)
            boxes.append((x1, y1, x2, y2))
    return boxes

with open(TEST_TXT, "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

saved = 0
for image_path in image_paths:
    if saved >= SAVE_COUNT:
        break

    label_path = os.path.splitext(image_path)[0] + ".txt"
    image = cv2.imread(image_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    boxes = load_yolo_boxes(label_path, w, h)
    if len(boxes) == 0:
        continue

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "traffic_light", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = os.path.join(OUT_DIR, f"gt_sample_{saved:03d}.jpg")
    cv2.imwrite(out_path, image)
    saved += 1

print("saved =", saved)
print("out_dir =", os.path.abspath(OUT_DIR))
