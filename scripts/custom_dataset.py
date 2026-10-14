
from pathlib import Path
from collections import defaultdict
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent

SOURCE_IMAGES = (
    ROOT / "dataset" / "rdd2022" /
    "RDD_SPLIT" / "test" / "images"
)
SOURCE_LABELS = (
    ROOT / "dataset" / "rdd2022" /
    "RDD_SPLIT" / "test" / "labels"
)

OUTPUT = ROOT / "dataset" / "road_test_200"
OUTPUT_IMAGES = OUTPUT / "images"
OUTPUT_LABELS = OUTPUT / "labels"

SAMPLE_SIZE = 200
CLEAN_IMAGE_TARGET = 40
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def valid_classes(label_path):
    """Return class IDs from valid YOLO annotation lines."""
    classes = set()

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return classes

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue

        if (
            class_id in range(5)
            and all(0 <= v <= 1 for v in (x, y, w, h))
            and w > 0 and h > 0
            and x - w / 2 >= 0
            and x + w / 2 <= 1
            and y - h / 2 >= 0
            and y + h / 2 <= 1
        ):
            classes.add(class_id)

    return classes


def main():
    if not SOURCE_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Images folder not found: {SOURCE_IMAGES}"
        )

    if not SOURCE_LABELS.is_dir():
        raise FileNotFoundError(
            f"Labels folder not found: {SOURCE_LABELS}"
        )

    rng = random.Random(SEED)

    class_candidates = defaultdict(list)
    clean_candidates = []
    seen_images = set()

    for image in SOURCE_IMAGES.iterdir():
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = SOURCE_LABELS / f"{image.stem}.txt"

        # Require a matching label file so that the subset
        # has a label file for every selected image.
        if not label.is_file():
            continue

        classes = valid_classes(label)

        if classes:
            for class_id in classes:
                class_candidates[class_id].append(
                    (image, label)
                )
        else:
            # Empty or invalid-only labels are retained as
            # potential clean examples for later inspection.
            clean_candidates.append((image, label))

    for items in class_candidates.values():
        rng.shuffle(items)

    rng.shuffle(clean_candidates)

    selected = []
    selected_stems = set()

    def add_item(item):
        image, label = item

        if image.stem in selected_stems:
            return False

        selected.append(item)
        selected_stems.add(image.stem)
        return True

    # Reserve up to 40 slots for images with no valid boxes.
    clean_count = min(CLEAN_IMAGE_TARGET, len(clean_candidates))

    for item in clean_candidates[:clean_count]:
        add_item(item)

    # Round-robin sampling encourages class representation.
    while len(selected) < SAMPLE_SIZE:
        added = False

        for class_id in range(5):
            for item in class_candidates[class_id]:
                if add_item(item):
                    added = True
                    break

            if len(selected) >= SAMPLE_SIZE:
                break

        if not added:
            break

    if len(selected) < SAMPLE_SIZE:
        print(
            f"Warning: only {len(selected)} eligible images "
            f"were found; requested {SAMPLE_SIZE}."
        )

    # Replace only this generated output folder.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    OUTPUT_IMAGES.mkdir(parents=True)
    OUTPUT_LABELS.mkdir(parents=True)

    class_image_counts = {i: 0 for i in range(5)}
    images_without_valid_boxes = 0

    for image, label in selected:
        shutil.copy2(image, OUTPUT_IMAGES / image.name)
        shutil.copy2(label, OUTPUT_LABELS / label.name)

        classes = valid_classes(label)

        if not classes:
            images_without_valid_boxes += 1

        for class_id in classes:
            class_image_counts[class_id] += 1

    yaml_text = f"""path: '{OUTPUT.as_posix()}'
train: images
val: images
test: images

names:
  0: class_0
  1: class_1
  2: class_2
  3: class_3
  4: class_4
"""

    (OUTPUT / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8"
    )

    print("\nDataset preparation complete.")
    print("Total selected:", len(selected))
    print("Images without valid boxes:", images_without_valid_boxes)
    print("Images containing each class:", class_image_counts)
    print("Output folder:", OUTPUT)
    print("Original dataset was not modified.")
    print(
        "\nNote: this script filters annotations, not visual content. "
        "Review images before interpreting them as clean-road examples."
    )


if __name__ == "__main__":
    main()
