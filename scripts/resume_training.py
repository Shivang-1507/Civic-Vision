
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "runs"
    / "civicvision"
    / "road_damage_5class"
    / "weights"
    / "last.pt"
)

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# Only inspect a checkpoint you trust from your own training run.
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint:", CHECKPOINT)
print("Available keys:", list(checkpoint.keys()))
print("Saved epoch:", checkpoint.get("epoch"))
print("Optimizer state available:", checkpoint.get("optimizer") is not None)
print("Training arguments available:", checkpoint.get("train_args") is not None)
print("Best fitness:", checkpoint.get("best_fitness"))

if checkpoint.get("optimizer") is not None and checkpoint.get("epoch") is not None:
    print("\nA true resume may be possible.")
else:
    print("\nThis checkpoint does not contain the required resume state.")
