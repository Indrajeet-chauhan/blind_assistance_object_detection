from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"  # Small pretrained model; downloads on first run.

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(MODEL_NAME)
        self.confidence_threshold = 0.40

    def detect(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=640,
            verbose=False
        )[0]
        annotated = result.plot()
        objects = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = self.model.names.get(class_id, str(class_id))
                objects.append({
                    "label": label,
                    "confidence": round(confidence, 3)
                })
        # Remove repeated labels in the spoken/list summary, keeping the highest confidence.
        best = {}
        for obj in objects:
            if obj["label"] not in best or obj["confidence"] > best[obj["label"]]["confidence"]:
                best[obj["label"]] = obj
        objects = sorted(best.values(), key=lambda x: x["confidence"], reverse=True)
        return annotated, objects
