
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

# Start from the best weights of your completed run
BEST_CHECKPOINT = (
    ROOT / "runs" / "civicvision" /
    "road_damage_5class" / "weights" / "best.pt"
)

PROJECT_DIR = ROOT / "runs" / "civicvision"

# =====================================================
# FINE-TUNING SETTINGS
# =====================================================

EPOCHS = 10
IMAGE_SIZE = 320
BATCH_SIZE = 16
WORKERS = 2
DEVICE = 0

LEARNING_RATE = 0.001
FINAL_LR_FACTOR = 0.1

RUN_NAME = "5_category_road_damage_finetune_1"

# =====================================================
# MAIN
# =====================================================

def main():
    print("Checking paths...", flush=True)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {BEST_CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Starting checkpoint:", BEST_CHECKPOINT)
    print("Dataset:", DATA_YAML)
    print("Epochs:", EPOCHS)
    print("Initial learning rate:", LEARNING_RATE)

    # Load the previously trained five-class model
    model = YOLO(str(BEST_CHECKPOINT))

    print("\nStarting fine-tuning...", flush=True)

    results = model.train(
        task="detect",
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,

        # Lower learning rate for fine-tuning
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        lrf=FINAL_LR_FACTOR,
        weight_decay=0.0005,

        # Training safeguards
        amp=True,
        cache=False,
        patience=5,
        seed=42,
        deterministic=True,

        # Save checkpoints and evaluation plots
        save=True,
        save_period=1,
        plots=True,
        val=True,

        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
    )

    print("\nFine-tuning completed.")
    print("Results directory:")
    print(PROJECT_DIR / RUN_NAME)

    print("Best checkpoint:")
    print(
        PROJECT_DIR / RUN_NAME /
        "weights" / "best.pt"
    )

    print("Last checkpoint:")
    print(
        PROJECT_DIR / RUN_NAME /
        "weights" / "last.pt"
    )


if __name__ == "__main__":
    main()
