from ultralytics import YOLO
import json

# Load your trained YOLOv8 model
model = YOLO("yolo_solar/train_results/weights/best.pt")

# Path to your single image
image_path = "dataset/test/images/testimg06.png"

# Dummy function to estimate PV area and capacity
def estimate_pv_properties(bboxes):
    panel_count = len(bboxes)
    area_sqm = panel_count * 1.5  # assume each panel ~1.5 m²
    capacity_kw = panel_count * 0.35  # assume each panel ~350W
    return panel_count, area_sqm, capacity_kw

# Run prediction
results = model.predict(image_path)
result = results[0]  # single image result

bboxes = result.boxes.xyxy.tolist()  # bounding boxes
confidences = result.boxes.conf.tolist()  # confidence scores

panel_count, pv_area_sqm, capacity_kw = estimate_pv_properties(bboxes)

output_data = {
    "has_solar": True if panel_count > 0 else False,
    "confidence": max(confidences) if confidences else 0.0,
    "panel_count_Est": panel_count,
    "pv_area_sqm_est": round(pv_area_sqm, 2),
    "capacity_kw_est": round(capacity_kw, 2),
    "qc_status": "verifiable" if panel_count > 0 else "not_detected"
}

# Print JSON output
print(json.dumps(output_data, indent=4))
