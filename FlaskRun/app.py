from flask import Flask, render_template, request, redirect, url_for, jsonify, session
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

#Flask Set up
app = Flask(__name__, template_folder="FrontEnd")
app.secret_key = "pothole-secret-key"

#Data Base setup for Flask
from Backend.editTable import connectDatabase, main
main()

#Folder declarations
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
    FRAME_SKIP = 1

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
        results = model_potholes(frame, conf=0.15, imgsz=512, verbose=False)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            # Add padding around the detected pothole box
            padding = 40    
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(frame.shape[1], x2 + padding)
            y2_padded = min(frame.shape[0], y2 + padding)

            # Use padded crop for classification
            pothole_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]


            if pothole_crop.shape[0] < 32 or pothole_crop.shape[1] < 32:
                continue

            crop_rgb = cv2.cvtColor(pothole_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_classification(input_tensor)
                # Convert outputs to probabilities
                probabilities = torch.softmax(outputs, dim=1)[0]
                # Get winning class
                confidence, predicted = torch.max(probabilities, 0)
                road_class_name = model_classification.classes[predicted.item()]
                road_confidence = confidence.item()
                # Print all class confidences
                print("\n---------------------------")
                print(f"Detected Class: {road_class_name}")
                print(f"Confidence: {road_confidence:.2f}")
                for i, class_name in enumerate(model_classification.classes):
                    print(f"{class_name}: {probabilities[i].item():.4f}")




            

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                f"{road_class_name} {road_confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            crop_filename = f"{base_name}_f{frame_index}_c{crop_index}.jpg"
            crop_save_path = os.path.join(app.config["CROPS_FOLDER"], crop_filename)
            cv2.imwrite(crop_save_path, pothole_crop)
            crop_paths.append({
                "path": f"crops/{crop_filename}",
                "class": road_class_name
            })
            crop_index += 1

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Tell the frontend we're now encoding
    tasks[task_id]["status"] = "encoding"
    tasks[task_id]["progress"] = 0

    ffmpeg_path = "ffmpeg"

    # Re-encode with FFmpeg, parsing progress from stderr
    process = subprocess.Popen([
        ffmpeg_path, "-y",
        "-i", raw_output_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
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

###################################
#           FLASK ROUTES   
###################################

@app.route("/")
def HomePage():
    return render_template("Home.html")


@app.route("/videoProcess")
def videoProcess():

    if "username" not in session:
        return redirect(url_for("LoginPage"))

    return render_template(
        "VideoProcessPage.html",
        username=session["username"],
        processed_video=None,
        crop_paths=[],
        original_video=None,
        processing=False,
        task_id=None,
        saved_videos=None
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/Login")
def LoginPage():
    return render_template("Login.html")

@app.route("/processedVideos")
def processedVideos():

    if "username" not in session:
        return redirect(url_for("LoginPage"))

    connection = connectDatabase()
    cursor = connection.cursor()

    cursor.execute("""
    SELECT video_name, processed_video_path, created_at
    FROM videos
    WHERE username = ?
    ORDER BY created_at DESC
    """, (session["username"],))

    saved_videos = cursor.fetchall()

    connection.close()

    return render_template(
        "VideoProcessPage.html",
        username=session["username"],
        processed_video=None,
        crop_paths=[],
        original_video=None,
        processing=False,
        task_id=None,
        saved_videos=saved_videos
    )

@app.route("/savedVideo/<video_name>")
def savedVideo(video_name):

    if "username" not in session:
        return redirect(url_for("LoginPage"))

    connection = connectDatabase()
    cursor = connection.cursor()

    cursor.execute("""
    SELECT processed_video_path
    FROM videos
    WHERE video_name = ? AND username = ?
    """, (video_name, session["username"]))

    video = cursor.fetchone()

    if not video:
        connection.close()
        return "Video not found", 404

    cursor.execute("""
    SELECT photo_path, classification
    FROM video_photos
    WHERE video_name = ?
    """, (video_name,))

    photos = cursor.fetchall()

    connection.close()

    crop_paths = [
        {
            "path": photo[0],
            "class": photo[1]
        }
        for photo in photos
    ]

    return render_template(
        "VideoProcessPage.html",
        username=session["username"],
        processed_video=video[0],
        crop_paths=crop_paths,
        original_video=None,
        processing=False,
        task_id=None,
        saved_videos=None
    )

#########################################################

@app.route("/upload", methods=["POST"])
def upload_video():
    print("Upload route hit")

    # Make sure user is logged in
    if "username" not in session:
        return redirect(url_for("LoginPage"))

    if "video" not in request.files:
        return redirect(url_for("HomePage"))

    file = request.files["video"]

    if file.filename == "":
        return redirect(url_for("HomePage"))

    filename = file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "username": session["username"],
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
        "VideoProcessPage.html",
        username=tasks[task_id]["username"],
        processed_video=None,
        crop_paths=[],
        original_video=tasks[task_id]["original_video"],
        processing=True,
        task_id=task_id,
        saved_videos=None
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

    username = tasks[task_id]["username"]
    video_name = task_id
    original_video = tasks[task_id]["original_video"]
    processed_video = tasks[task_id]["processed_video"]
    crop_paths = tasks[task_id]["crop_paths"]

    # Save processed video to database
    connection = connectDatabase()
    cursor = connection.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO videos (
        video_name,
        username,
        original_video_path,
        processed_video_path
    )
    VALUES (?, ?, ?, ?)
    """, (video_name, username, original_video, processed_video))

    for crop in crop_paths:
        cursor.execute("""
        INSERT INTO video_photos (
            video_name,
            photo_path,
            classification
        )
        VALUES (?, ?, ?)
        """, (
            video_name,
            crop["path"],
            crop["class"]
        ))

    connection.commit()
    connection.close()

    return render_template(
        "VideoProcessPage.html",
        username=username,
        processed_video=processed_video,
        crop_paths=crop_paths,
        original_video=original_video,
        processing=False,
        task_id=None,
        saved_videos=None
    )


###########################
#   Login 
###########################
@app.route("/login", methods=["POST"])
def login():

    username = request.form.get("username")
    password = request.form.get("password")

    connection = connectDatabase()
    cursor = connection.cursor()

    cursor.execute("""
    SELECT *
    FROM users
    WHERE username = ? AND password = ?
    """, (username, password))

    user = cursor.fetchone()

    connection.close()

    if user:
        session["username"] = username
        return redirect(url_for("videoProcess"))

    return render_template(
        "Login.html",
        login_error="Invalid username or password"
    )

###########################
#   Sign Up
###########################
@app.route("/signup", methods=["POST"])
def signup():

    # Get form data
    username = request.form.get("username")
    password = request.form.get("password")

    # Connect to database
    connection = connectDatabase()
    cursor = connection.cursor()

    # Check If Username Exists
    cursor.execute("""
    SELECT *
    FROM users
    WHERE username = ?
    """, (username,))
    existingUser = cursor.fetchone()


    # Username already exists
    if existingUser:
        connection.close()
        return render_template(
            "Login.html",
            signup_error="Username already exists"
        )


    # Create New User if Username not Taken
    cursor.execute("""
    INSERT INTO users (username, password)
    VALUES (?, ?)
    """, (username, password))


    # Save Changes and Close Database
    connection.commit()
    print("User created successfully.")
    connection.close()

    # Save Session and Redirect User
    session["username"] = username
    return redirect(url_for("videoProcess"))

###########################
#   Logout
###########################
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("HomePage"))

###########################
#   Main
###########################
if __name__ == "__main__":

    # Run database setup first
    print("Setting up SQL Database")
    main()
    print()

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)