
from pathlib import Path
import sys
import torch
from ultralytics import YOLO

# ============================================
# CIVICVISION: IMPROVED FINE-TUNING
# ============================================

ROOT = Path(__file__).resolve().parent.parent

BEST_MODEL = (
    ROOT / "runs" / "civicvision"
    / "pothole_final_50" / "weights" / "best.pt"
)

DATA_YAML = ROOT / "dataset" / "pothole_yolo" / "data.yaml"
OUTPUT_DIR = ROOT / "runs" / "civicvision"

# Updated training settings
EPOCHS = 20
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 2
DEVICE = 0
LEARNING_RATE = 0.015


def main():
    if not BEST_MODEL.is_file():
        sys.exit(f"Best checkpoint not found:\n{BEST_MODEL}")

    if not DATA_YAML.is_file():
        sys.exit(f"Dataset configuration not found:\n{DATA_YAML}")

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU unavailable. Check your PyTorch setup.")

    print("CivicVision: improved fine-tuning")
    print("Starting model:", BEST_MODEL)
    print("Dataset:", DATA_YAML)
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    model = YOLO(str(BEST_MODEL))

    results = model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        lr0=LEARNING_RATE,
        lrf=0.015,
        optimizer="AdamW",
        weight_decay=0.0005,
        patience=10,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        save_period=1,
        val=True,
        seed=0,
        project=str(OUTPUT_DIR),
        name="pothole_finetune_640_lr0015",
        exist_ok=False,
    )

    print("\nFine-tuning completed.")
    print("Results directory:", results.save_dir)
    print("Best model:", Path(results.save_dir) / "weights" / "best.pt")
    print("Last model:", Path(results.save_dir) / "weights" / "last.pt")


if __name__ == "__main__":
    main()
