
from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================
# CIVICVISION: FINAL MODEL TESTING
# ============================================================

# Script is inside CivicVision/scripts/
ROOT = Path(__file__).resolve().parent.parent

RUN_DIR = (
    ROOT
    / "runs"
    / "civicvision"
    / "road_damage_5class_finetune_final"
)

MODEL_PATH = RUN_DIR / "weights" / "best.pt"

DATA_YAML = ROOT / "dataset" / "rdd2022" / "data_five_class.yaml"

TEST_IMAGES = (
    ROOT / "dataset" / "rdd2022" / "RDD_SPLIT" / "test" / "images"
)

OUTPUT_DIR = ROOT / "runs" / "civicvision" / "final_test"

IMAGE_SIZE = 320
BATCH_SIZE = 8
WORKERS = 2
DEVICE = 0
CONFIDENCE = 0.25


def main():
    print("=" * 60)
    print("CIVICVISION FINAL MODEL TEST")
    print("=" * 60)
    print("Project root:", ROOT)

    # --------------------------------------------------------
    # 1. Verify paths
    # --------------------------------------------------------
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not MODEL_PATH.is_file():
        print("\nExpected checkpoint not found:")
        print(MODEL_PATH)

        weights_root = ROOT / "runs" / "civicvision"

        if weights_root.exists():
            checkpoints = list(weights_root.rglob("*.pt"))

            print("\nAvailable model checkpoints:")
            for checkpoint in checkpoints:
                print(" -", checkpoint.relative_to(ROOT))
        else:
            print("\nCivicVision runs directory not found:", weights_root)

        raise FileNotFoundError(
            "\nSet MODEL_PATH to the correct fine-tuned CivicVision checkpoint."
        )

    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Test images directory not found:\n{TEST_IMAGES}"
        )

    # --------------------------------------------------------
    # 2. Verify GPU
    # --------------------------------------------------------
    print("\nCUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    print("GPU:", torch.cuda.get_device_name(DEVICE))

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------
    print("\nLoading model:", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))

    print("Model classes:", model.names)

    # --------------------------------------------------------
    # 4. Evaluate on held-out test split
    # --------------------------------------------------------
    print("\nStarting test evaluation...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        plots=True,
        project=str(OUTPUT_DIR),
        name="evaluation",
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 5. Print metrics
    # --------------------------------------------------------
    box = metrics.box

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)

    print(f"Precision : {box.mp:.4f} ({box.mp * 100:.2f}%)")
    print(f"Recall    : {box.mr:.4f} ({box.mr * 100:.2f}%)")
    print(f"mAP50     : {box.map50:.4f} ({box.map50 * 100:.2f}%)")
    print(f"mAP50-95  : {box.map:.4f} ({box.map * 100:.2f}%)")

    print("\nPer-class AP:")
    for class_id, ap50, ap in zip(
        box.ap_class_index, box.ap50, box.ap
    ):
        print(
            f"{model.names[int(class_id)]}: "
            f"AP50={ap50:.4f}, AP50-95={ap:.4f}"
        )

    # --------------------------------------------------------
    # 6. Save sample predictions
    # --------------------------------------------------------
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    test_files = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not test_files:
        raise RuntimeError("No test images found.")

    print("\nSaving predictions for up to 20 test images...")

    model.predict(
        source=[str(p) for p in test_files[:20]],
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name="sample_predictions",
        exist_ok=True,
    )

    print("\nTesting completed.")
    print("Evaluation outputs:", OUTPUT_DIR / "evaluation")
    print("Prediction images:", OUTPUT_DIR / "sample_predictions")


if __name__ == "__main__":
    main()
