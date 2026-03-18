cd /mnt/c/Users/akshi/Downloads/611/Assignment3

cat > submission/README.md <<'MD'
# Assignment 3 - Small Object Detection Using YOLO

## Overview
In this assignment, I performed small object detection for traffic lights using YOLOv8n on the Bosch Small Traffic Lights Dataset (BSTLD). I first evaluated the pretrained YOLOv8n model as a baseline, then fine-tuned the model on a one-class version of BSTLD where all traffic light states were merged into a single class called `traffic_light`.

## Dataset
- Dataset used: Bosch Small Traffic Lights Dataset (BSTLD)
- Original annotation format: YAML
- Converted format: YOLO detection format
- Class setup: all traffic light states collapsed into one class
- Final split:
  - Train: 3565 images
  - Validation: 763 images
  - Test: 765 images
- Bounding boxes:
  - Train: 7525
  - Validation: 1588
  - Test: 1643

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy torch torchvision torchaudio
Main Scripts

test_yolo.py
Verified that YOLOv8n could run correctly in the environment.

prepare_bstld_yolo_oneclass.py
Converted the BSTLD YAML labels into YOLO text labels and created the train/val/test split.

baseline_pretrained_bstld.py
Evaluated the pretrained YOLOv8n model on the BSTLD test split.

eval_trained_bstld.py
Evaluated the fine-tuned YOLOv8n model on the BSTLD test split.

sweep_conf_nms.py
Swept confidence threshold and NMS IoU values to find a better inference setting.

save_gt_samples.py
Saved ground-truth annotated dataset sample images.

save_final_predictions.py
Saved final predicted output images using the best confidence/NMS setting.

Training

I fine-tuned yolov8n.pt on the BSTLD one-class dataset for 15 epochs at image size 640.

Results
Pretrained baseline

Precision: 0.5040

Recall: 0.4553

F1: 0.4784

Fine-tuned model

Precision: 0.7737

Recall: 0.6847

F1: 0.7265

Improvement after fine-tuning

Precision improvement: +0.2697

Recall improvement: +0.2295

F1 improvement: +0.2481

Confidence and NMS Sweep

I tested multiple confidence thresholds and NMS IoU settings.

Best balanced setting:

Confidence threshold: 0.25

NMS IoU: 0.45

F1: 0.7489

Highest precision setting:

Confidence threshold: 0.40

NMS IoU: 0.45

Precision: 0.9017

Submission Contents

code/ — Python scripts used in the project

deliverables/trained_model/ — trained YOLO model weights

deliverables/annotated_dataset_samples/ — example ground-truth labeled images

deliverables/baseline_results/ — pretrained baseline metrics and samples

deliverables/trained_results/ — trained model metrics and threshold/NMS sweep outputs

deliverables/final_predictions/ — final predicted result images

deliverables/training_artifacts/ — training outputs such as CSV logs and sample training images

deliverables/dataset_info/ — dataset YAML and summary file

report/ — final report PDF
MD


Verify:

```bash id="t9p6e5"
sed -n '1,220p' submission/README.md