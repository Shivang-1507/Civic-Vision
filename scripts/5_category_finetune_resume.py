
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CIVICVISION: CONTINUE FROM AN EARLIER FINE-TUNE 1 CHECKPOINT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    ROOT / "dataset" / "rdd2022" / "data_five_class.yaml"
)

WEIGHTS_DIR = (
    ROOT
    / "runs"
    / "civicvision"
    / "5_category_road_damage_finetune_1"
    / "weights"
)

# Select an earlier epoch checkpoint.
# Change to "best.pt" if that checkpoint performs better.
CHECKPOINT = WEIGHTS_DIR / "epoch9.pt"

PROJECT = ROOT / "runs" / "civicvision"
RUN_NAME = "5_category_road_damage_epoch9_finetune"

# Training settings
EPOCHS = 10
IMAGE_SIZE = 320
BATCH_SIZE = 8
WORKERS = 0
LEARNING_RATE = 0.0005

# ============================================================
# VALIDATE PATHS
# ============================================================

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

print("=" * 60)
print("CIVICVISION: NEW FINE-TUNING RUN")
print("=" * 60)
print(f"Dataset: {DATA_YAML}")
print(f"Starting checkpoint: {CHECKPOINT}")
print(f"New epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60)

# Load the selected trained five-class model
model = YOLO(str(CHECKPOINT))

# Explicitly use the CivicVision dataset.
# This is a NEW fine-tuning run, not a resume operation.
model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    workers=WORKERS,
    device=0,
    optimizer="AdamW",
    lr0=LEARNING_RATE,
    lrf=0.1,
    weight_decay=0.0005,
    patience=10,
    amp=True,
    cache=False,
    seed=42,
    deterministic=True,
    project=str(PROJECT),
    name=RUN_NAME,
    exist_ok=False,
    save=True,
    save_period=1,
    plots=True,
    val=True,
)

print("\nFine-tuning finished.")
print(f"Results directory: {PROJECT / RUN_NAME}")
print(f"Best model: {PROJECT / RUN_NAME / 'weights' / 'best.pt'}")
