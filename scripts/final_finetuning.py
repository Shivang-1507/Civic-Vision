
from pathlib import Path
import torch
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "rdd2022" / "RDD_SPLIT"
DATA_YAML = ROOT / "dataset" / "rdd2022" / "data_five_class.yaml"

# Original trained five-class model
MODEL = (
    ROOT
    / "runs"
    / "civicvision"
    / "road_damage_5class"
    / "weights"
    / "best.pt"
)

# Training configuration
EPOCHS = 100
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

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    if not MODEL.is_file():
        raise FileNotFoundError(
            f"Original model checkpoint not found: {MODEL}"
        )

    print("\nCUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(DEVICE))
    print("\nLoading original five-class model:", MODEL)

    model = YOLO(str(MODEL))

    if model.model.yaml.get("nc") != 5:
        raise ValueError(
            "The selected checkpoint is not a five-class model."
        )

    print("\nStarting five-class fine-tuning...")

    model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,

        # Fine-tuning learning rate
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,

        # Performance and caching
        amp=True,
        cache=False,

        # Training controls
        patience=5,
        save=True,
        save_period=1,
        plots=True,
        val=True,
        seed=0,

        # Separate output folder
        project=str(ROOT / "runs" / "civicvision"),
        name="road_damage_5class_finetune_final",
        exist_ok=False
    )

    output = (
        ROOT
        / "runs"
        / "civicvision"
        / "road_damage_5class_finetune_original_lr"
    )

    print("\nFine-tuning completed.")
    print("Best model:", output / "weights" / "best.pt")
    print("Last checkpoint:", output / "weights" / "last.pt")


if __name__ == "__main__":
    main()
