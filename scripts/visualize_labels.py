
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent

image_path = ROOT / "dataset" / "pothole_yolo" / "images" / "test" / "India_004060.jpg"
label_path = ROOT / "dataset" / "pothole_yolo" / "labels" / "test" / "India_004060.txt"

print("Image exists:", image_path.is_file())
print("Label exists:", label_path.is_file())

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

height, width = image.shape[:2]

for line in label_path.read_text().splitlines():
    values = line.split()

    if len(values) != 5:
        continue

    class_id = int(values[0])
    x_center, y_center, box_width, box_height = map(float, values[1:])

    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"Class {class_id}",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

output_path = Path("results/annotated_example.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), image)

print(f"Saved annotated image to: {output_path}")
