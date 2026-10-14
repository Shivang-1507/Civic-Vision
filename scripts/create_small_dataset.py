
from pathlib import Path
import random
import shutil
import sys

# ============================================
# CIVICVISION: CREATE A SMALLER DATASET
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE = PROJECT_ROOT / "dataset" / "pothole_yolo"
DESTINATION = PROJECT_ROOT / "dataset" / "pothole_yolo_small"

# Number of images to select from each split
TARGET_COUNTS = {
    "train": 3000,
    "val": 600,
    "test": 600,
}

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_image_label_pairs(split):
    image_dir = SOURCE / "images" / split
    label_dir = SOURCE / "labels" / split

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images or labels for split: {split}"
        )

    pairs = []

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing label for image: {image_path.name}"
            )

        # Non-empty label = at least one annotated pothole
        has_pothole = bool(label_path.read_text().strip())

        pairs.append((image_path, label_path, has_pothole))

    return pairs


def select_images(pairs, target_count, split):
    if len(pairs) < target_count:
        raise ValueError(
            f"{split} contains only {len(pairs)} images; "
            f"{target_count} requested."
        )

    positives = [p for p in pairs if p[2]]
    negatives = [p for p in pairs if not p[2]]

    # Preserve approximately the original positive/negative ratio
    positive_target = round(
        target_count * len(positives) / len(pairs)
    )
    negative_target = target_count - positive_target

    if positive_target > len(positives):
        positive_target = len(positives)
        negative_target = target_count - positive_target

    if negative_target > len(negatives):
        negative_target = len(negatives)
        positive_target = target_count - negative_target

    selected = (
        random.sample(positives, positive_target)
        + random.sample(negatives, negative_target)
    )

    random.shuffle(selected)

    print(
        f"{split}: selected {len(selected)} images | "
        f"pothole images: {positive_target} | "
        f"negative images: {negative_target}"
    )

    return selected


def main():
    random.seed(RANDOM_SEED)

    if not SOURCE.is_dir():
        sys.exit(f"Source dataset not found:\n{SOURCE}")

    # Avoid accidentally overwriting an existing dataset
    if DESTINATION.exists():
        sys.exit(
            f"Destination already exists:\n{DESTINATION}\n"
            "Rename or remove that folder before running again."
        )

    print("Creating CivicVision small dataset...")
    print(f"Source: {SOURCE}")
    print(f"Destination: {DESTINATION}\n")

    try:
        for split, target_count in TARGET_COUNTS.items():
            pairs = get_image_label_pairs(split)

            print(f"\nProcessing {split}...")
            selected = select_images(
                pairs, target_count, split
            )

            output_images = DESTINATION / "images" / split
            output_labels = DESTINATION / "labels" / split

            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)

            for image_path, label_path, _ in selected:
                shutil.copy2(
                    image_path,
                    output_images / image_path.name
                )
                shutil.copy2(
                    label_path,
                    output_labels / label_path.name
                )

        # Write the Colab-compatible/local YAML configuration
        yaml_content = f"""path: {DESTINATION.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: pothole
"""

        (DESTINATION / "data.yaml").write_text(
            yaml_content, encoding="utf-8"
        )

    except Exception:
        # Remove an incomplete subset if creation fails
        if DESTINATION.exists():
            shutil.rmtree(DESTINATION)
        raise

    print("\nDataset creation completed successfully!")
    print(f"Dataset folder: {DESTINATION}")
    print(f"Configuration: {DESTINATION / 'data.yaml'}")
    print("\nOriginal dataset has not been modified.")


if __name__ == "__main__":
    main()
