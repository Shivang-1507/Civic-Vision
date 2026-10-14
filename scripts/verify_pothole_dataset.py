
from pathlib import Path

ROOT = Path("dataset/pothole_yolo")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val", "test"]:
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    labels = list(labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    invalid_lines = 0
    wrong_classes = 0
    invalid_boxes = 0
    positive_images = 0
    boxes = 0

    for label in labels:
        has_pothole = False

        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()

            if not values:
                continue

            if len(values) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(values[0])
                x, y, w, h = map(float, values[1:])
            except ValueError:
                invalid_lines += 1
                continue

            if class_id != 0:
                wrong_classes += 1

            if (
                not all(map(lambda v: 0 <= v <= 1, [x, y, w, h]))
                or w <= 0 or h <= 0
                or x - w / 2 < -1e-6
                or y - h / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6
                or y + h / 2 > 1 + 1e-6
            ):
                invalid_boxes += 1

            if class_id == 0:
                has_pothole = True
                boxes += 1

        if has_pothole:
            positive_images += 1

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images missing labels:", len(missing_labels))
    print("Labels missing images:", len(missing_images))
    print("Images with potholes:", positive_images)
    print("Pothole boxes:", boxes)
    print("Invalid annotation lines:", invalid_lines)
    print("Wrong class IDs:", wrong_classes)
    print("Invalid bounding boxes:", invalid_boxes)
