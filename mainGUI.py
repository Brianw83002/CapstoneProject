import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

# ----------------------------
# Load YOLO model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_potholes = YOLO("pothole.pt")
model_potholes.to(device)

# ----------------------------
# Create input/output folders
# ----------------------------
INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "output_videos"

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# Globals
# ----------------------------
cap = None
out = None
playing_cap = None
total_frames = 0
current_frame = 0
processing = False

# ----------------------------
# Refresh file lists
# ----------------------------
def refresh_lists():
    input_list.delete(0, tk.END)
    output_list.delete(0, tk.END)

    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".mp4"):
            input_list.insert(tk.END, file)

    for file in os.listdir(OUTPUT_FOLDER):
        if file.endswith(".mp4"):
            output_list.insert(tk.END, file)

# ----------------------------
# Process selected input video
# ----------------------------
def process_selected_video():
    global cap, out, total_frames, current_frame, processing

    selected = input_list.curselection()
    if not selected:
        return

    filename = input_list.get(selected[0])
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")

    cap = cv2.VideoCapture(input_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    progress_bar["value"] = 0
    processing = True

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    process_frame()

# ----------------------------
# Frame processing
# ----------------------------
def process_frame():
    global cap, out, current_frame, processing

    if not processing:
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        progress_bar["value"] = 100
        processing = False
        refresh_lists()
        print("✅ Processing complete!")
        return

    current_frame += 1
    progress = (current_frame / total_frames) * 100
    progress_bar["value"] = progress

    annotated_frame = frame.copy()

    results = model_potholes(frame, conf=0.2, verbose=False)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    display_frame(annotated_frame)
    out.write(annotated_frame)

    window.after(1, process_frame)

# ----------------------------
# Play video (input or output)
# ----------------------------
def play_selected_video(folder):
    global playing_cap

    if folder == "input":
        selected = input_list.curselection()
        if not selected:
            return
        filename = input_list.get(selected[0])
        path = os.path.join(INPUT_FOLDER, filename)

    else:
        selected = output_list.curselection()
        if not selected:
            return
        filename = output_list.get(selected[0])
        path = os.path.join(OUTPUT_FOLDER, filename)

    playing_cap = cv2.VideoCapture(path)
    play_frame()

def play_frame():
    global playing_cap

    if playing_cap is None:
        return

    ret, frame = playing_cap.read()
    if not ret:
        playing_cap.release()
        return

    display_frame(frame)
    window.after(30, play_frame)

# ----------------------------
# Display frame in GUI
# ----------------------------
def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 360))  # Resize for GUI
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# ----------------------------
# GUI
# ----------------------------
window = tk.Tk()
window.title("Pothole Detection Manager")
window.geometry("1000x600")

main_frame = tk.Frame(window)
main_frame.pack(fill="both", expand=True)

# LEFT PANEL (File Lists)
left_panel = tk.Frame(main_frame)
left_panel.pack(side="left", fill="y", padx=10, pady=10)

tk.Label(left_panel, text="Input Videos").pack()
input_list = tk.Listbox(left_panel, width=30, height=15)
input_list.pack()

tk.Button(left_panel, text="Process Selected",
          command=process_selected_video).pack(pady=5)

tk.Button(left_panel, text="Watch Input",
          command=lambda: play_selected_video("input")).pack(pady=5)

tk.Label(left_panel, text="Output Videos").pack(pady=(20, 0))
output_list = tk.Listbox(left_panel, width=30, height=15)
output_list.pack()

tk.Button(left_panel, text="Watch Output",
          command=lambda: play_selected_video("output")).pack(pady=5)

# RIGHT PANEL (Video Display)
right_panel = tk.Frame(main_frame)
right_panel.pack(side="right", fill="both", expand=True)

video_label = tk.Label(right_panel)
video_label.pack(pady=10)

progress_bar = ttk.Progressbar(
    right_panel,
    orient="horizontal",
    length=600,
    mode="determinate"
)
progress_bar.pack(pady=10)

refresh_lists()

window.mainloop()