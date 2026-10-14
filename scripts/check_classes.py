
from pathlib import Path
from collections import Counter
import time

labels_dir = Path("dataset/rdd2022/RDD_SPLIT/train/labels")

start = time.time()
class_counts = Counter()
empty_files = 0
invalid_lines = 0
total_files = 0

for label_file in labels_dir.glob("*.txt"):
    total_files += 1

    try:
        with label_file.open("r", encoding="utf-8") as f:
            lines = f

            found_object = False
            for line in lines:
                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:
                    invalid_lines += 1
                    continue

                try:
                    class_id = int(values[0])
                    class_counts[class_id] += 1
                    found_object = True
                except ValueError:
                    invalid_lines += 1

            if not found_object:
                empty_files += 1

    except OSError as error:
        print(f"Could not read {label_file.name}: {error}")

    if total_files % 5000 == 0:
        print(f"Processed {total_files} files...")

print("\n--- SUMMARY ---")
print("Label files processed:", total_files)
print("Class counts:", dict(sorted(class_counts.items())))
print("Empty label files:", empty_files)
print("Invalid annotation lines:", invalid_lines)
print(f"Time taken: {time.time() - start:.2f} seconds")
