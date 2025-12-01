import os
import torch
from torchvision import transforms
from PIL import Image
from road_classification import RoadDetectionCNN

# ----------------------------
# RUN main2.py FIRST
# ----------------------------


# ----------------------------
# Settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crop_folder = "debug_crops"

# ----------------------------
# Load CNN model
# ----------------------------
model_classification = RoadDetectionCNN(num_classes=4)
model_classification.load_state_dict(torch.load("road_classification_model.pth", map_location=device))
model_classification.eval()
model_classification.to(device)
model_classification.classes = ['D00', 'D10', 'D20', 'D40']

# ----------------------------
# Define transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------------------
# Run CNN on folder
# ----------------------------
class_counts = {cls: 0 for cls in model_classification.classes}

image_files = [f for f in os.listdir(crop_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(crop_folder, img_file)
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_classification(input_tensor)
        _, predicted = torch.max(outputs, 1)
        cls_name = model_classification.classes[predicted.item()]
        class_counts[cls_name] += 1

# ----------------------------
# Print results
# ----------------------------
print("CNN classification results for debug_crops:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} images")
