from ultralytics import YOLO

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data="dataset/data.yaml",   # path to your YAML file
        epochs=50,                  # adjust as needed
        imgsz=640,
        batch=8,
        device="cuda",
        project="yolo_solar",
        name="train_results",
        exist_ok=True
    )

    print("✅ Training complete! Model saved in yolo_solar/train_results/weights/best.pt")


if __name__ == "__main__":
    main()
