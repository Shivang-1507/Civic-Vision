
"""
CivicVision - Second Five-Class Road Damage Fine-Tuning
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# =====================================================
# PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    ROOT / "dataset" / "rdd2022" /
    "data_five_class.yaml"
)

# Best checkpoint from the FIRST fine-tuning run
BEST_CHECKPOINT = (
    ROOT / "runs" / "civicvision" /
    "5_category_road_damage_finetune_1" /
    "weights" / "best.pt"
)

PROJECT_DIR = ROOT / "runs" / "civicvision"

# =====================================================
# TRAINING SETTINGS
# =====================================================

EPOCHS = 100
IMAGE_SIZE = 320
BATCH_SIZE = 16
WORKERS = 2
DEVICE = 0

LEARNING_RATE = 0.001
FINAL_LR_FACTOR = 0.1
WEIGHT_DECAY = 0.0005

# SECOND fine-tuning run
RUN_NAME = "5_category_road_damage_finetune_2"


def main():
    print("Checking paths...", flush=True)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "First fine-tuning checkpoint not found:\n"
            f"{BEST_CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Check your PyTorch installation."
        )

    print("\nCivicVision: Second Fine-Tuning Run")
    print("Starting checkpoint:", BEST_CHECKPOINT)
    print("Dataset:", DATA_YAML)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Workers:", WORKERS)
    print("Initial learning rate:", LEARNING_RATE)
    print("Run name:", RUN_NAME)

    # Load the first fine-tuned model
    model = YOLO(str(BEST_CHECKPOINT))

    print("\nStarting second fine-tuning run...", flush=True)

    results = model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,

        # Fine-tuning optimizer and learning rate
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        lrf=FINAL_LR_FACTOR,
        weight_decay=WEIGHT_DECAY,

        # Training safeguards
        amp=True,
        cache=False,
        patience=5,
        seed=42,
        deterministic=True,

        # Checkpoints and evaluation
        save=True,
        save_period=1,
        plots=True,
        val=True,

        # Preserve the first run
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
    )

    run_dir = Path(results.save_dir)

    print("\nSecond fine-tuning run completed.")
    print("Results directory:", run_dir)
    print("Best checkpoint:", run_dir / "weights" / "best.pt")
    print("Last checkpoint:", run_dir / "weights" / "last.pt")


if __name__ == "__main__":
    main()
