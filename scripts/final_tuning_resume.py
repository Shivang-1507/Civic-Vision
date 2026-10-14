
from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================
# CIVICVISION: RESUME FINAL FIVE-CLASS FINE-TUNING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RUN_DIR = (
    ROOT
    / "runs"
    / "civicvision"
    / "road_damage_5class_finetune_final"
)

LAST_CHECKPOINT = RUN_DIR / "weights" / "epoch53.pt"


def main():
    # --------------------------------------------------------
    # Verify checkpoint
    # --------------------------------------------------------
    if not LAST_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Last checkpoint not found:\n{LAST_CHECKPOINT}\n"
            "Make sure the fine-tuning run has started and saved a checkpoint."
        )

    # --------------------------------------------------------
    # Verify CUDA
    # --------------------------------------------------------
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Resume checkpoint:", LAST_CHECKPOINT)

    # --------------------------------------------------------
    # Load and resume the existing training run
    # --------------------------------------------------------
    model = YOLO(str(LAST_CHECKPOINT))

    print("\nAttempting to resume the existing fine-tuning run...")

    model.train(resume=True)

    print("\nResume training completed.")


if __name__ == "__main__":
    main()
