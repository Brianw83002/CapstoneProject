from ultralytics import YOLO

# Load base model
model = YOLO("yolov8n.pt")

# Train on pothole dataset
model.train(data="Pothole-detection-YOLOv8.v1i.yolov8/data.yaml", epochs=100, imgsz=640)

# Save trained weights
model.save("pothole.pt")