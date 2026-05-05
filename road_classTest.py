import torch
import cv2
import glob
import torchvision.transforms as transforms
from Model.road_classification import RoadDetectionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["D00", "D10", "D20", "D40"]

model = RoadDetectionCNN(num_classes=4)
model.load_state_dict(torch.load("Model/road_classification_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Automatically load all jpg images
image_paths = glob.glob("test_images/*.jpg")

for image_path in image_paths:

    print(f"\nLoading: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)

    print(f"Predicted: {classes[predicted.item()]}")
    print(f"Confidence: {confidence.item():.2f}")

    for i, class_name in enumerate(classes):
        print(f"{class_name}: {probabilities[i].item():.4f}")