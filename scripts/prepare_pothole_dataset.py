
from pathlib import Path
import shutil

# Original dataset location
SOURCE = Path("dataset/rdd2022/RDD_SPLIT")

# Destination for the pothole-only dataset
DEST = Path("dataset/pothole_yolo")

# Class 4 is assumed to represent potholes.
POTHOLE_CLASS_ID = 4

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(images_dir, image_stem):
    """Find an image even if its extension is not .jpg."""
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def prepare_split(split):
    images_dir = SOURCE / split / "images"
    labels_dir = SOURCE / split / "labels"

    output_images = DEST / "images" / split
    output_labels = DEST / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total_images = 0
    positive_images = 0
    pothole_boxes = 0
    negative_images = 0
    missing_images = 0

    for label_path in labels_dir.glob("*.txt"):
        image_path = find_image(images_dir, label_path.stem)

        if image_path is None:
            missing_images += 1
            continue

        retained_labels = []

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            values = line.split()

            if not values:
                continue

            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation: {label_path}"
                )

            class_id = int(values[0])

            if class_id == POTHOLE_CLASS_ID:
                # Remap the original class 4 to our class 0.
                retained_labels.append(
                    "0 " + " ".join(values[1:])
                )
                pothole_boxes += 1

        # Copy each image and write its filtered labels.
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        (output_labels / f"{label_path.stem}.txt").write_text(
            "\n".join(retained_labels),
            encoding="utf-8"
        )

        total_images += 1

        if retained_labels:
            positive_images += 1
        else:
            negative_images += 1

    print(f"\n{split.upper()} SUMMARY")
    print("-" * 35)
    print("Images copied:", total_images)
    print("Images with potholes:", positive_images)
    print("Images without pothole labels:", negative_images)
    print("Pothole bounding boxes:", pothole_boxes)
    print("Labels with missing images:", missing_images)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        prepare_split(split)

    print("\nDataset preparation finished.")
