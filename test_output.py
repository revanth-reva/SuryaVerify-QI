from ultralytics import YOLO

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = "yolo_solar/train_results/weights/best.pt"  # your trained model
IMAGE_PATH = "dataset/test/images/testimg01.png"            # image to test
PIXEL_SCALE_M = 0.2   # meters per pixel (adjust based on satellite zoom)
PANEL_EFFICIENCY = 0.18  # 18% typical PV efficiency
SOLAR_IRRADIANCE = 1.0   # kW/m² (standard test conditions)

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO(MODEL_PATH)

# -------------------------
# RUN INFERENCE
# -------------------------
results = model.predict(source=IMAGE_PATH, conf=0.25)  # adjust confidence as needed

# -------------------------
# EXTRACT BOUNDING BOXES
# -------------------------
bboxes = []
for r in results:
    if hasattr(r, 'boxes'):
        for box in r.boxes.xyxy.cpu().numpy():  # get bounding box in pixels [x_min, y_min, x_max, y_max]
            bboxes.append(box)

# -------------------------
# CALCULATE AREA AND CAPACITY
# -------------------------
total_area_sqm = 0
for bbox in bboxes:
    x_min, y_min, x_max, y_max = bbox
    width_m = (x_max - x_min) * PIXEL_SCALE_M
    height_m = (y_max - y_min) * PIXEL_SCALE_M
    total_area_sqm += width_m * height_m

capacity_kw = total_area_sqm * SOLAR_IRRADIANCE * PANEL_EFFICIENCY
panel_count_est = len(bboxes)

# -------------------------
# CREATE OUTPUT JSON
# -------------------------
output = {
    "has_solar": panel_count_est > 0,
    "confidence": float(results[0].boxes.conf.max().cpu()) if panel_count_est > 0 else 0.0,
    "panel_count_Est": panel_count_est,
    "pv_area_sqm_est": round(total_area_sqm, 2),
    "capacity_kw_est": round(capacity_kw, 2),
    "qc_status": "verifiable" if panel_count_est > 0 else "not_detected"
}

print(output)
