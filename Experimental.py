import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
from road_classification import RoadDetectionCNN
import os

# ----------------------------
# Settings
# ----------------------------
input_video = "potholeVideo.mp4"
output_video = "potholeVid_output.mp4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Load YOLO pothole model
# ----------------------------
model_potholes = YOLO("pothole.pt")
model_potholes.to(device)

# ----------------------------
# Load road classification model
# ----------------------------
model_classification = RoadDetectionCNN(num_classes=4)
model_classification.load_state_dict(torch.load("road_classification_model.pth", map_location=device))
model_classification.eval()
model_classification.to(device)
model_classification.classes = ['D00', 'D10', 'D20', 'D40']

# ----------------------------
# Define transforms for CNN
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------------------
# Video setup
# ----------------------------
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# ----------------------------
# Create folder to save cropped patches
# ----------------------------
os.makedirs("debug_crops", exist_ok=True)

# ----------------------------
# Process video
# ----------------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    annotated_frame = frame.copy()

    # ---- YOLO pothole detection ----
    results_potholes = model_potholes(frame, conf=0.2, verbose=False)
    for i, box in enumerate(results_potholes[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # ---- Enlarge tiny boxes automatically ----
        w, h = x2 - x1, y2 - y1
        min_size = 64  # minimum width/height for CNN input
        if w < min_size:
            dw = (min_size - w) // 2
            x1 = max(x1 - dw, 0)
            x2 = min(x2 + dw, frame.shape[1])
        if h < min_size:
            dh = (min_size - h) // 2
            y1 = max(y1 - dh, 0)
            y2 = min(y2 + dh, frame.shape[0])

        pothole_crop = frame[y1:y2, x1:x2]
        if pothole_crop.shape[0] < 32 or pothole_crop.shape[1] < 32:
            continue

        # ---- Resize crop to CNN input ----
        crop_rgb = cv2.cvtColor(pothole_crop, cv2.COLOR_BGR2RGB)
        input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

        # ---- CNN classification ----
        with torch.no_grad():
            outputs = model_classification(input_tensor)
            _, predicted = torch.max(outputs, 1)
            road_class_name = model_classification.classes[predicted.item()]

        # ---- Draw bounding box + class ----
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated_frame,
            road_class_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # ---- Save crop for debugging ----
        crop_resized = cv2.resize(pothole_crop, (128, 128))
        crop_filename = f"debug_crops/frame{frame_count}_box{i}.png"
        cv2.imwrite(crop_filename, crop_resized)

    # ---- Write output frame ----
    out.write(annotated_frame)

cap.release()
out.release()
print(f"✅ Processed video saved as {output_video}")
print("✅ Cropped patches saved in debug_crops folder")
