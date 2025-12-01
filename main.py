import cv2
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
from road_classification import RoadDetectionCNN  # your CNN class

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
model_potholes = YOLO("pothole.pt")  # your trained YOLO pothole model
model_potholes.to(device)

# ----------------------------
# Load road classification model
# ----------------------------
model_classification = RoadDetectionCNN(num_classes=4)
model_classification.load_state_dict(torch.load("road_classification_model.pth", map_location=device))
model_classification.eval()
model_classification.to(device)

# Define road class names (match your dataset)
model_classification.classes = ['D00', 'D10', 'D20', 'D40']

# ----------------------------
# Define transforms for CNN
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # match training input size
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
# Process video
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # ---- Road classification for this frame ----
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_classification(input_tensor)
        _, predicted = torch.max(outputs, 1)
        road_class_name = model_classification.classes[predicted.item()]

    # ---- YOLO pothole detection ----
    results_potholes = model_potholes(frame, conf=0.2, verbose=False)
    for i, box in enumerate(results_potholes[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Draw box (red)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Overlay road classification instead of confidence
        cv2.putText(
            annotated_frame,
            f"Road: {road_class_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),  # green text
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
