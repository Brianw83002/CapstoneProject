import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import torchvision.transforms as transforms
from road_classification import RoadDetectionCNN

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Load YOLO model
# ----------------------------
model_potholes = YOLO("potholeV2.pt")
model_potholes.to(device)

# ----------------------------
# Load road classification model
# ----------------------------
model_classification = RoadDetectionCNN(num_classes=4)
model_classification.load_state_dict(
    torch.load("road_classification_model.pth", map_location=device)
)
model_classification.eval()
model_classification.to(device)
model_classification.classes = ['D00', 'D10', 'D20', 'D40']

# ----------------------------
# CNN Transform
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ----------------------------
# Output folder
# ----------------------------
OUTPUT_FOLDER = "output_videos"
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
video_paths = []

display_paused = False
last_frame = None
last_crops = []  # Store crops to display

# ----------------------------
# Browse videos
# ----------------------------
def browse_videos():
    global video_paths
    files = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if files:
        video_paths = list(files)
        refresh_lists()

# ----------------------------
# Refresh lists
# ----------------------------
def refresh_lists():
    input_list.delete(0, tk.END)
    output_list.delete(0, tk.END)

    for path in video_paths:
        filename = os.path.basename(path)
        input_list.insert(tk.END, filename)

    if os.path.exists(OUTPUT_FOLDER):
        for file in os.listdir(OUTPUT_FOLDER):
            if file.endswith(".mp4"):
                output_list.insert(tk.END, file)

# ----------------------------
# Process selected video
# ----------------------------
def process_selected_video():
    global cap, out, total_frames, current_frame, processing

    selected = input_list.curselection()
    if not selected:
        return

    input_path = video_paths[selected[0]]
    filename = os.path.basename(input_path)
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
    global cap, out, current_frame, processing, last_frame, last_crops

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
    last_crops = []

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
            2,
        )

        # Store crop to display
        crop_img = cv2.resize(pothole_crop, (80, 80))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        last_crops.append(ImageTk.PhotoImage(Image.fromarray(crop_img)))

    # Store latest frame
    last_frame = annotated_frame.copy()

    # Update video display if not paused
    if not display_paused:
        display_frame(annotated_frame)
        display_crops()

    out.write(annotated_frame)
    window.after(1, process_frame)

# ----------------------------
# Display crops
# ----------------------------
def display_crops():
    # Clear previous crops
    for widget in crops_container.winfo_children():
        widget.destroy()
    # Display current crops
    for img in last_crops:
        lbl = tk.Label(crops_container, image=img)
        lbl.image = img  # Keep reference
        lbl.pack(side="left", padx=2)

# ----------------------------
# Pause / Resume display
# ----------------------------
def toggle_pause():
    global display_paused
    display_paused = not display_paused
    if display_paused:
        pause_button.config(text="▶ Resume")
    else:
        pause_button.config(text="⏸ Pause")
        if last_frame is not None:
            display_frame(last_frame)
            display_crops()

# ----------------------------
# Video playback (input/output)
# ----------------------------
def play_selected_video(folder):
    global playing_cap
    if folder == "input":
        selected = input_list.curselection()
        if not selected:
            return
        path = video_paths[selected[0]]
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
# Display frame
# ----------------------------
def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 360))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

# ----------------------------
# GUI
# ----------------------------
window = tk.Tk()
window.title("Pothole Detection Manager")
window.geometry("1000x750")

main_frame = tk.Frame(window)
main_frame.pack(fill="both", expand=True)

left_panel = tk.Frame(main_frame)
left_panel.pack(side="left", fill="y", padx=10, pady=10)

tk.Label(left_panel, text="Input Videos").pack()
tk.Button(left_panel, text="Browse Videos", command=browse_videos).pack(pady=5)
input_list = tk.Listbox(left_panel, width=30, height=15)
input_list.pack()
tk.Button(left_panel, text="Process Selected", command=process_selected_video).pack(pady=5)
tk.Button(left_panel, text="Watch Input", command=lambda: play_selected_video("input")).pack(pady=5)
tk.Label(left_panel, text="Output Videos").pack(pady=(20, 0))
output_list = tk.Listbox(left_panel, width=30, height=15)
output_list.pack()
tk.Button(left_panel, text="Watch Output", command=lambda: play_selected_video("output")).pack(pady=5)

right_panel = tk.Frame(main_frame)
right_panel.pack(side="right", fill="both", expand=True)

video_label = tk.Label(right_panel)
video_label.pack(pady=10)

progress_bar = ttk.Progressbar(
    right_panel, orient="horizontal", length=600, mode="determinate"
)
progress_bar.pack(pady=10)

pause_button = tk.Button(right_panel, text="⏸ Pause", command=toggle_pause)
pause_button.pack(pady=5)

# Scrollable frame for crops
frame_crops_panel = tk.Frame(right_panel)
frame_crops_panel.pack(fill="x", pady=10)
crops_canvas = tk.Canvas(frame_crops_panel, height=100)
crops_scroll = ttk.Scrollbar(frame_crops_panel, orient="horizontal", command=crops_canvas.xview)
crops_canvas.configure(xscrollcommand=crops_scroll.set)
crops_scroll.pack(side="bottom", fill="x")
crops_canvas.pack(side="top", fill="x")
crops_container = tk.Frame(crops_canvas)
crops_canvas.create_window((0, 0), window=crops_container, anchor="nw")

def update_crops_scrollregion(event):
    crops_canvas.configure(scrollregion=crops_canvas.bbox("all"))
crops_container.bind("<Configure>", update_crops_scrollregion)

window.mainloop()