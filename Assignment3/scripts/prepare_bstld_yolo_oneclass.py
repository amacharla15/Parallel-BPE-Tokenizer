import os
import random
import yaml
from PIL import Image

RAW_ROOT = "data/raw/bstld/dataset_train_rgb"
LABEL_YAML = "external/bstld/label_files/train.yaml"
OUT_DIR = "data/processed/bstld_oneclass"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clip01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def convert_box(box, img_w, img_h):
    x_min = float(box["x_min"])
    x_max = float(box["x_max"])
    y_min = float(box["y_min"])
    y_max = float(box["y_max"])

    bw = x_max - x_min
    bh = y_max - y_min

    if bw <= 0 or bh <= 0:
        return None

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    cx = clip01(cx / img_w)
    cy = clip01(cy / img_h)
    bw = clip01(bw / img_w)
    bh = clip01(bh / img_h)

    if bw <= 0 or bh <= 0:
        return None

    return (cx, cy, bw, bh)

def main():
    ensure_dir(OUT_DIR)

    with open(LABEL_YAML, "r") as f:
        records = yaml.safe_load(f)

    rng = random.Random(SEED)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:]
    }

    summary_lines = []
    summary_lines.append(f"total_images: {n}")
    summary_lines.append(f"train_images: {len(splits['train'])}")
    summary_lines.append(f"val_images: {len(splits['val'])}")
    summary_lines.append(f"test_images: {len(splits['test'])}")

    for split_name in ["train", "val", "test"]:
        split_txt_path = os.path.join(OUT_DIR, f"{split_name}.txt")
        total_boxes_split = 0
        kept_images = 0

        with open(split_txt_path, "w") as split_f:
            for item in splits[split_name]:
                rel_path = item["path"]
                if rel_path.startswith("./"):
                    rel_path = rel_path[2:]

                img_path = os.path.join(RAW_ROOT, rel_path)
                if not os.path.isfile(img_path):
                    continue

                kept_images += 1

                label_path = os.path.splitext(img_path)[0] + ".txt"

                with Image.open(img_path) as im:
                    img_w, img_h = im.size

                yolo_lines = []
                for box in item.get("boxes", []):
                    out = convert_box(box, img_w, img_h)
                    if out is None:
                        continue
                    cx, cy, bw, bh = out
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    total_boxes_split += 1

                with open(label_path, "w") as lf:
                    if yolo_lines:
                        lf.write("\n".join(yolo_lines) + "\n")
                    else:
                        lf.write("")

                split_f.write(os.path.abspath(img_path) + "\n")

        summary_lines.append(f"{split_name}_kept_images: {kept_images}")
        summary_lines.append(f"{split_name}_boxes: {total_boxes_split}")

    dataset_yaml = {
        "path": os.path.abspath(OUT_DIR),
        "train": os.path.abspath(os.path.join(OUT_DIR, "train.txt")),
        "val": os.path.abspath(os.path.join(OUT_DIR, "val.txt")),
        "test": os.path.abspath(os.path.join(OUT_DIR, "test.txt")),
        "names": {
            0: "traffic_light"
        }
    }

    with open(os.path.join(OUT_DIR, "dataset.yaml"), "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print("DONE")
    print("\n".join(summary_lines))
    print("dataset_yaml:", os.path.abspath(os.path.join(OUT_DIR, "dataset.yaml")))

if __name__ == "__main__":
    main()
