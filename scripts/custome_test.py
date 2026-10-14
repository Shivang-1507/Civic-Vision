
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "civicvision" /
    "road_damage_5class" / "weights" / "best.pt"
)

DATA_YAML = (
    ROOT / "dataset" / "road_test_200" / "data.yaml"
)

OUTPUT_DIR = ROOT / "runs" / "civicvision"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Test dataset YAML not found: {DATA_YAML}\n"
            "Run prepare_road_test_200.py first."
        )

    print("Loading trained model...", flush=True)
    model = YOLO(str(MODEL_PATH))

    print("Evaluating on the prepared road dataset...", flush=True)

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=512,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name="road_test_200_evaluation",
        exist_ok=True
    )

    print("\n===== 200-IMAGE EVALUATION RESULTS =====")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    print(f"mAP@50:     {results.box.map50:.4f}")
    print(f"mAP@50-95:  {results.box.map:.4f}")

    print("\nResults saved to:")
    print(OUTPUT_DIR / "road_test_200_evaluation")


if __name__ == "__main__":
    main()
