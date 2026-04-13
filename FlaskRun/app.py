from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
import sys
import threading
import uuid
import subprocess

# opens Model Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))
from Model.road_classification import RoadDetectionCNN

app = Flask(__name__, template_folder="FrontEnd")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
CROPS_FOLDER = "static/crops"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["CROPS_FOLDER"] = CROPS_FOLDER

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Load models once
# ----------------------------
model_potholes = YOLO("../Model/potholeV2.pt")
model_potholes.to(device)

model_classification = RoadDetectionCNN(num_classes=4)
model_classification.load_state_dict(
    torch.load("../Model/road_classification_model.pth", map_location=device)
)
model_classification.eval()
model_classification.to(device)
model_classification.classes = ['D00', 'D10', 'D20', 'D40']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# stores active tasks
tasks = {}


def process_video(input_path, filename, task_id):
    FRAME_SKIP = 2

    base_name = os.path.splitext(filename)[0]
    raw_output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"raw_{base_name}.avi")
    web_output_filename = f"processed_{base_name}.mp4"
    web_output_path = os.path.join(app.config["OUTPUT_FOLDER"], web_output_filename)
    crop_paths = []

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    # Write to AVI with XVID as a reliable intermediate format
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    frame_index = 0
    crop_index = 0

    tasks[task_id]["status"] = "processing"
    tasks[task_id]["progress"] = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        tasks[task_id]["progress"] = int((frame_index / total_frames) * 100)

        if frame_index % FRAME_SKIP != 0:
            out.write(frame)
            continue

        annotated_frame = frame.copy()
        results = model_potholes(frame, conf=0.2, verbose=False)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            pothole_crop = frame[y1:y2, x1:x2]
            if pothole_crop.shape[0] < 32 or pothole_crop.shape[1] < 32:
                continue

            crop_rgb = cv2.cvtColor(pothole_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_classification(input_tensor)
                _, predicted = torch.max(outputs, 1)
                road_class_name = model_classification.classes[predicted.item()]

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                road_class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            crop_filename = f"{base_name}_f{frame_index}_c{crop_index}.jpg"
            crop_save_path = os.path.join(app.config["CROPS_FOLDER"], crop_filename)
            cv2.imwrite(crop_save_path, pothole_crop)
            crop_paths.append(f"crops/{crop_filename}")
            crop_index += 1

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Tell the frontend we're now encoding
    tasks[task_id]["status"] = "encoding"
    tasks[task_id]["progress"] = 0

    # Re-encode with FFmpeg, parsing progress from stderr
    process = subprocess.Popen([
        "ffmpeg", "-y",
        "-i", raw_output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        web_output_path
    ], stderr=subprocess.PIPE, text=True)

    for line in process.stderr:
        if "frame=" in line:
            try:
                encoded_frame = int(line.split("frame=")[1].strip().split()[0])
                encode_progress = min(int((encoded_frame / total_frames) * 100), 99)
                tasks[task_id]["progress"] = encode_progress
            except (ValueError, IndexError):
                pass

    process.wait()

    if process.returncode != 0:
        print("FFmpeg encoding failed")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error"] = "FFmpeg re-encoding failed"
        return

    # Clean up the raw intermediate file
    if os.path.exists(raw_output_path):
        os.remove(raw_output_path)

    tasks[task_id]["status"] = "done"
    tasks[task_id]["progress"] = 100
    tasks[task_id]["processed_video"] = f"outputs/{web_output_filename}"
    tasks[task_id]["crop_paths"] = crop_paths


@app.route("/")
def home():
    return render_template(
        "HomePage.html",
        processed_video=None,
        crop_paths=[],
        original_video=None,
        processing=False,
        task_id=None
    )


@app.route("/upload", methods=["POST"])
def upload_video():
    print("Upload route hit")

    if "video" not in request.files:
        return redirect(url_for("home"))

    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("home"))

    filename = file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "original_video": f"uploads/{filename}",
        "processed_video": None,
        "crop_paths": []
    }

    thread = threading.Thread(target=process_video, args=(save_path, filename, task_id))
    thread.start()

    return redirect(url_for("processing_page", task_id=task_id))


@app.route("/processing/<task_id>")
def processing_page(task_id):
    if task_id not in tasks:
        return "Task not found", 404

    return render_template(
        "HomePage.html",
        processed_video=None,
        crop_paths=[],
        original_video=tasks[task_id]["original_video"],
        processing=True,
        task_id=task_id
    )


@app.route("/status/<task_id>")
def check_status(task_id):
    if task_id not in tasks:
        return jsonify({"status": "not_found"}), 404

    return jsonify({
        "status": tasks[task_id]["status"],
        "progress": tasks[task_id]["progress"]
    })


@app.route("/result/<task_id>")
def result_page(task_id):
    if task_id not in tasks:
        return "Task not found", 404

    if tasks[task_id]["status"] == "error":
        return f"Processing failed: {tasks[task_id].get('error', 'Unknown error')}", 500

    if tasks[task_id]["status"] != "done":
        return redirect(url_for("processing_page", task_id=task_id))

    print("Processed video path:", tasks[task_id]["processed_video"])
    print(
        "Full file exists:",
        os.path.exists(os.path.join("static", tasks[task_id]["processed_video"]))
    )

    return render_template(
        "HomePage.html",
        processed_video=tasks[task_id]["processed_video"],
        crop_paths=tasks[task_id]["crop_paths"],
        original_video=tasks[task_id]["original_video"],
        processing=False,
        task_id=None
    )


if __name__ == "__main__":
    app.run(debug=True)