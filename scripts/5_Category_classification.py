from pathlib import Path
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "rdd2022" / "RDD_SPLIT"
DATA_YAML = ROOT / "dataset" / "rdd2022" / "data_five_class.yaml"
MODEL = ROOT / "yolov8n.pt"

# Training configuration
EPOCHS = 50
IMAGE_SIZE = 320
BATCH_SIZE = 16
WORKERS = 2
DEVICE = 0

def main():
    # Verify dataset directories
    for split in ["train", "val", "test"]:
        images = DATASET / split / "images"
        labels = DATASET / split / "labels"

        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(
                f"Missing images or labels directory for: {split}"
            )

        image_count = sum(
            1 for p in images.iterdir()
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        )
        print(f"{split}: {image_count} images")

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL}\n"
            "Place yolov8n.pt in your CivicVision project root."
        )

    # Configure the five-class dataset
    DATA_YAML.write_text(
        f"""path: '{DATASET.as_posix()}'
train: train/images
val: val/images
test: test/images

names:
  0: class_0
  1: class_1
  2: class_2
  3: class_3
  4: class_4
""",
        encoding="utf-8"
    )

    print("\nCUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Check your PyTorch installation.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("\nStarting five-class YOLOv8n training...")

    model = YOLO(str(MODEL))

    model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        optimizer="auto",
        lr0=0.01,
        amp=True,
        cache=False,
        patience=5,
        save=True,
        save_period=1,
        plots=True,
        val=True,
        seed=0,
        project=str(ROOT / "runs" / "civicvision"),
        name="road_damage_5class",
        exist_ok=False
    )

    print("\nTraining completed.")
    print(
        "Best model:",
        ROOT / "runs" / "civicvision" /
        "road_damage_5class" / "weights" / "best.pt"
    )
    print(
        "Last checkpoint:",
        ROOT / "runs" / "civicvision" /
        "road_damage_5class" / "weights" / "last.pt"
    )


if __name__ == "__main__":
    main()