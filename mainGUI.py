import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# ----------------------------
# Globals
# ----------------------------
cap = None
out = None
running = False
model_potholes = None
model_ready = False

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model in background
# ----------------------------
def load_model():
    global model_potholes, model_ready
    print("⏳ Loading YOLO model...")
    model_potholes = YOLO("pothole.pt")
    model_potholes.to(device)
    model_ready = True
    print("✅ Model loaded")

# ----------------------------
# Video processing
# ----------------------------
def process_frame():
    global cap, out, running

    if not running or cap is None or not model_ready:
        window.after(10, process_frame)
        return

    ret, frame = cap.read()
    if not ret:
        running = False
        cap.release()
        out.release()
        print("✅ Processed video saved as output_processed.mp4")
        return

    annotated_frame = frame.copy()

    results = model_potholes(frame, conf=0.2, verbose=False)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(img)

    video_label.configure(image=imgtk)
    video_label.imgtk = imgtk

    out.write(annotated_frame)

    window.after(1, process_frame)

# ----------------------------
# Select video
# ----------------------------
def select_video():
    global cap, out, running

    if not model_ready:
        status_label.config(text="Loading model, please wait...")
        return

    file_path = filedialog.askopenfilename(
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )

    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_processed.mp4", fourcc, fps, (width, height))

    status_label.config(text="Processing...")
    running = True
    process_frame()

# ----------------------------
# GUI (CREATED FIRST)
# ----------------------------
window = tk.Tk()
window.title("Pothole Detection")

status_label = tk.Label(window, text="Loading model...")
status_label.pack(pady=5)

select_btn = tk.Button(window, text="Select Video", command=select_video)
select_btn.pack(pady=10)

video_label = tk.Label(window)
video_label.pack()

# ----------------------------
# Start model loading AFTER GUI
# ----------------------------
threading.Thread(target=load_model, daemon=True).start()

window.mainloop()