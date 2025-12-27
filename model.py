from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

# warm-up (critical)
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model(dummy, imgsz=640, verbose=False)

def predict(img):
    results = model(img, imgsz=640, conf=0.25, verbose=False)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0])
        })

    return {
        "count": len(detections),
        "detections": detections
    }
