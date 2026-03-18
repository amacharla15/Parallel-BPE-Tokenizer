import os
from ultralytics import YOLO

TEST_TXT = "data/processed/bstld_oneclass/test.txt"
WEIGHTS = "runs/detect/results/train/yolov8n_bstld_640/weights/best.pt"
OUT_DIR = "results/final_predictions_best"
CONF = 0.25
NMS_IOU = 0.45
IMG_SIZE = 640
SAVE_COUNT = 30

os.makedirs(OUT_DIR, exist_ok=True)

with open(TEST_TXT, "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

model = YOLO(WEIGHTS)

saved = 0
for image_path in image_paths:
    if saved >= SAVE_COUNT:
        break

    results = model.predict(
        source=image_path,
        conf=CONF,
        iou=NMS_IOU,
        imgsz=IMG_SIZE,
        verbose=False,
        save=False
    )

    r = results[0]
    plotted = r.plot()

    out_path = os.path.join(OUT_DIR, f"pred_{saved:03d}.jpg")
    import cv2
    cv2.imwrite(out_path, plotted)
    saved += 1

print("saved =", saved)
print("out_dir =", os.path.abspath(OUT_DIR))
