from ultralytics import YOLO

# Load the trained model
model = YOLO("yolo_solar/train_results/weights/best.pt")

# Run inference on a single image
results = model("dataset/test/images/testimg07.png")

# Show results
results[0].plot(save=True)  # saves annotated image
