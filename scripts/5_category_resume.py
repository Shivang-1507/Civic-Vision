
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Must match the training run name
RUN_DIR = ROOT / "runs" / "civicvision" / "road_damage_5class"
LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"

def main():
    if not LAST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {LAST_CHECKPOINT}\n"
            "Start training first and ensure at least one checkpoint is saved."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Check your PyTorch installation."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Resuming from:", LAST_CHECKPOINT)

    model = YOLO(str(LAST_CHECKPOINT))

    model.train(
        resume=True,
        device=0,
        workers=2
    )

    print("Training finished or resumed run completed.")

if __name__ == "__main__":
    main()
