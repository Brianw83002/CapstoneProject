import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
from road_classification import RoadDetectionCNN  # your CNN class
from collections import deque

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
    transforms.Resize((128, 128)),  # match CNN training input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------------------
# Temporal smoothing setup
# ----------------------------
smooth_frames = 5  # number of frames to average over
pothole_history = {}  # {box_id: deque of previous predictions}

def get_box_id(box, threshold=20):
    """Assign an ID to a bounding box based on position proximity"""
    x1, y1, x2, y2 = box
    for existing_id, history in pothole_history.items():
        ex_x1, ex_y1, ex_x2, ex_y2 = history[-1]['box']
        if abs(x1 - ex_x1) < threshold and abs(y1 - ex_y1) < threshold:
            return existing_id
    return len(pothole_history) + 1

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
# Process video
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # ---- YOLO pothole detection ----
    results_potholes = model_potholes(frame, conf=0.2, verbose=False)
    for i, box in enumerate(results_potholes[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Add padding and skip tiny boxes
        pad = 20
        x1_p = max(x1 - pad, 0)
        y1_p = max(y1 - pad, 0)
        x2_p = min(x2 + pad, frame.shape[1])
        y2_p = min(y2 + pad, frame.shape[0])

        pothole_crop = frame[y1_p:y2_p, x1_p:x2_p]
        if pothole_crop.shape[0] < 32 or pothole_crop.shape[1] < 32:
            continue  # skip very small crops

        # Make crop square
        h, w, _ = pothole_crop.shape
        size = max(h, w)
        square_crop = 255 * np.ones((size, size, 3), dtype=np.uint8)
        square_crop[:h, :w] = pothole_crop

        # Prepare for CNN
        crop_rgb = cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)
        input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

        # ---- Road classification ----
        with torch.no_grad():
            outputs = model_classification(input_tensor)
            _, predicted = torch.max(outputs, 1)
            road_class_name = model_classification.classes[predicted.item()]

        # ---- Temporal smoothing ----
        box_id = get_box_id((x1, y1, x2, y2))
        if box_id not in pothole_history:
            pothole_history[box_id] = deque(maxlen=smooth_frames)
        pothole_history[box_id].append({'box': (x1, y1, x2, y2), 'class': road_class_name})

        # Most frequent class in history
        classes_in_history = [entry['class'] for entry in pothole_history[box_id]]
        smoothed_class = max(set(classes_in_history), key=classes_in_history.count)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Overlay smoothed road class
        cv2.putText(
            annotated_frame,
            smoothed_class,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ---- Write output frame ----
    out.write(annotated_frame)

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
print(f"✅ Processed video saved as {output_video}")
