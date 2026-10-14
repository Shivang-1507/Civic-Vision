
"""
CivicVision - Fresh Pothole Training with Resume Support
"""

from pathlib import Path
import sys
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "pothole_yolo" / "data.yaml"
PRETRAINED = ROOT / "yolov8n.pt"
OUTPUT = ROOT / "runs" / "civicvision"
RUN_NAME = "pothole_final_50"

# Training configuration
IMAGE_SIZE = 416
TOTAL_EPOCHS = 50
BATCH_SIZE = 4
WORKERS = 2
DEVICE = 0


def main():
    if not DATA_YAML.is_file():
        sys.exit(f"Dataset YAML not found: {DATA_YAML}")

    if not PRETRAINED.is_file():
        sys.exit(f"Pretrained weights not found: {PRETRAINED}")

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available. Check your PyTorch setup.")

    print("CivicVision: starting a NEW training run")
    print("Dataset:", DATA_YAML)
    print("Starting weights:", PRETRAINED)
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Epoch target: {TOTAL_EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}, batch: {BATCH_SIZE}")

    model = YOLO(str(PRETRAINED))

    results = model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=TOTAL_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(OUTPUT),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=False,
        plots=True,
        save=True,
        save_period=1,   # Save an epoch checkpoint every epoch
        val=True,
        seed=0,
    )

    print("\nTraining completed.")
    print("Run directory:", results.save_dir)
    print("Checkpoints:", Path(results.save_dir) / "weights")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Check the run's weights folder "
            "for an epoch checkpoint such as epoch4.pt."
        )
