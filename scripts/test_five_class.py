
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Use the best model saved during training
MODEL_PATH = (
    ROOT / "runs" / "civicvision" /
    "road_damage_5class" / "weights" / "best.pt"
)

# Original test dataset
DATA_YAML = ROOT / "dataset" / "rdd2022" / "data_five_class.yaml"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    model = YOLO(str(MODEL_PATH))

    print("Evaluating best.pt on the TEST dataset...")
    print("Model:", MODEL_PATH)
    print("Dataset:", DATA_YAML)

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=512,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / "runs" / "civicvision"),
        name="road_damage_5class_test",
        exist_ok=True
    )

    print("\n===== TEST RESULTS =====")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    print(f"mAP50:      {results.box.map50:.4f}")
    print(f"mAP50-95:   {results.box.map:.4f}")

    print("\nTest results saved in:")
    print(
        ROOT / "runs" / "civicvision" /
        "road_damage_5class_test"
    )

if __name__ == "__main__":
    main()
