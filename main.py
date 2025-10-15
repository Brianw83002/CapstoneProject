import cv2
from ultralytics import YOLO

# ----------------------------
# Video input/output settings
# ----------------------------
input_video = "TestVideos/dashcam_video.mp4"
output_video = "TestVideos/dashcam_video_Output.mp4"

# ----------------------------
# Load YOLOv8 models
# ----------------------------
model_objects = YOLO("yolov8n.pt")   # Default objects (cars, people, etc.)
model_potholes = YOLO("pothole.pt")  # Potholes (replace with your trained model)

# Force CPU usage
device = "cpu"
model_objects.to(device)
model_potholes.to(device)

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

    # ---- YOLO inference for objects ----
    results_objects = model_objects(frame, conf=0.5, verbose=False)
    for i, box in enumerate(results_objects[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        cls = int(results_objects[0].boxes.cls[i])
        conf = results_objects[0].boxes.conf[i]

        # Draw box (blue)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            annotated_frame,
            f"{model_objects.names[cls]} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # ---- YOLO inference for potholes ----
    results_potholes = model_potholes(frame, conf=0.5, verbose=False)
    for i, box in enumerate(results_potholes[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        conf = results_potholes[0].boxes.conf[i]

        # Draw box (red)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated_frame,
            f"Pothole {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
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
