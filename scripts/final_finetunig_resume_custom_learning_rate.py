
from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================
# CIVICVISION: RESUME FINE-TUNING WITH LOWER LEARNING RATE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RUN_DIR = (
    ROOT
    / "runs"
    / "civicvision"
    / "road_damage_5class_finetune_final"
)

LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"

# Resume configuration
NEW_LR = 0.0001
BATCH_SIZE = 16
IMAGE_SIZE = 320
WORKERS = 2
DEVICE = 0


def main():
    # --------------------------------------------------------
    # Verify checkpoint and CUDA
    # --------------------------------------------------------
    if not LAST_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{LAST_CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(DEVICE))
    print("Checkpoint:", LAST_CHECKPOINT)
    print("Target learning rate:", NEW_LR)

    model = YOLO(str(LAST_CHECKPOINT))

    # --------------------------------------------------------
    # Lower the learning rate after the trainer is restored.
    # This callback runs at the start of each resumed epoch.
    # --------------------------------------------------------
    def lower_lr_callback(trainer):
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = NEW_LR

        # Keep scheduler base rates consistent with the new LR
        if hasattr(trainer, "scheduler"):
            for base_lr in trainer.scheduler.base_lrs:
                pass
            trainer.scheduler.base_lrs = [
                NEW_LR for _ in trainer.optimizer.param_groups
            ]

        print(
            f"\nLearning rate set to {NEW_LR:.6f} "
            f"for epoch {trainer.epoch + 1}"
        )

    model.add_callback("on_train_epoch_start", lower_lr_callback)

    # --------------------------------------------------------
    # Resume the existing run
    # --------------------------------------------------------
    print("\nResuming existing fine-tuning run...")

    model.train(resume=True)

    print("\nResumed training finished.")


if __name__ == "__main__":
    main()
