import cv2
import torch
from ultralytics import YOLO
import numpy as np

#Videos
input_video = "TestVideos/dashcam_video.mp4"
output_video = "TestVideos/dashcam_video_Output.mp4"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # fast version
midas.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
midas.to(device)

# Transformation for MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # matches MiDaS_small

# Video input/output

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Approximate scaling factor for relative depth to feet (tune as needed)
SCALE_FEET = 50  # 1.0 (MiDaS farthest) -> ~50 feet

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=0.5, verbose=False)

    # Prepare frame for MiDaS
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    # Copy frame for manual annotation
    annotated_frame = frame.copy()

    # Normalize depth (0 = closest, 1 = farthest)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # Overlay YOLO detections and distances
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        box_depth_norm = np.median(depth_norm[y1:y2, x1:x2])

        # Convert relative depth to approximate feet
        box_distance_ft = SCALE_FEET * box_depth_norm

        # Color coding: Red (close) -> Green (far)
        r = int(255 * (1 - box_depth_norm))
        g = int(255 * box_depth_norm)
        b = 0
        color = (b, g, r)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Put distance in feet
        cv2.putText(
            annotated_frame,
            f"{box_distance_ft:.1f} ft",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Put class and confidence
        cls = int(results[0].boxes.cls[i])
        conf = results[0].boxes.conf[i]
        cv2.putText(
            annotated_frame,
            f"{model.names[cls]} {conf:.2f}",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    out.write(annotated_frame)

cap.release()
out.release()
print(f"✅ Processed video saved as {output_video}")
