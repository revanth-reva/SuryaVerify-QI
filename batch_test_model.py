import os
from ultralytics import YOLO

model = YOLO("yolo_solar/train_results/weights/best.pt")
test_folder = "dataset/test/images"

for img_file in os.listdir(test_folder):
    if img_file.endswith((".jpg", ".png")):
        img_path = os.path.join(test_folder, img_file)
        results = model(img_path)
        for r in results:
            r.save()  # Saves annotated images
