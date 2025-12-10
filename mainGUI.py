import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# ----------------------------
# Load YOLO model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_potholes = YOLO("pothole.pt")
model_potholes.to(device)

# ----------------------------
# Video variables
# ----------------------------
cap = None
out = None

# ----------------------------
# Video processing function
# ----------------------------
def process_frame():
    global cap, out
    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        print("✅ Processed video saved as output_processed.mp4")
        return

    annotated_frame = frame.copy()

    # YOLO pothole detection
    results_potholes = model_potholes(frame, conf=0.2, verbose=False)
    for box in results_potholes[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Convert frame to ImageTk
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    out.write(annotated_frame)

    # Schedule next frame
    window.after(1, process_frame)

# ----------------------------
# Select video
# ----------------------------
def select_video():
    global cap, out
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output_processed.mp4", fourcc, fps, (width, height))

        process_frame()

# ----------------------------
# GUI
# ----------------------------
window = tk.Tk()
window.title("Pothole Detection")

select_btn = tk.Button(window, text="Select Video", command=select_video)
select_btn.pack(pady=10)

video_label = tk.Label(window)
video_label.pack()

window.mainloop()

