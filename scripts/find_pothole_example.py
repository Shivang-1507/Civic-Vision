
from pathlib import Path
import cv2

root = Path("dataset/rdd2022/RDD_SPLIT/train")
images_dir = root / "images"
labels_dir = root / "labels"

for label_path in labels_dir.glob("*.txt"):
    lines = label_path.read_text(encoding="utf-8").splitlines()

    # Find an image whose labels include class 4.
    if not any(line.split() and line.split()[0] == "4" for line in lines):
        continue

    image_path = images_dir / f"{label_path.stem}.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    for line in lines:
        values = line.split()
        if len(values) != 5:
            continue

        class_id = int(values[0])
        xc, yc, bw, bh = map(float, values[1:])

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, f"Class {class_id}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    output = Path("results/pothole_example.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)

    print("Label file:", label_path.name)
    print("Image file:", image_path.name)
    print("Saved visualization:", output)
    break
else:
    print("No class-4 annotation found in the training labels.")
