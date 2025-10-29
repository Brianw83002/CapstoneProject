from ultralytics import YOLO
import torch

def main():
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Load current model
    model = YOLO("pothole.pt")

    # Train on pothole dataset with GPU optimization
    model.train(
        data="Pothole.v1-raw.yolov8/data.yaml", 
        epochs=100, 
        imgsz=640,
        device=0,              # Use GPU 0
        workers=0,             # Set workers to 0 to avoid multiprocessing issues
        batch=16,              # Batch size
        patience=10,           # Early stopping patience
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate
        momentum=0.937,        # SGD momentum
        weight_decay=0.0005,   # Optimizer weight decay
        warmup_epochs=3.0,     # Warmup epochs
        warmup_momentum=0.8,   # Warmup initial momentum
        save=True,             # Save train checkpoints
        exist_ok=True,         # Overwrite existing project/name
        pretrained=True,       # Use pretrained weights
        optimizer='auto',      # Optimizer selection
        verbose=True,          # Print results
        seed=42,               # Training seed
        amp=True,              # Automatic Mixed Precision
    )

    # Save trained weights
    model.save("potholeV2.pt")

    # Validate the model
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

    print("✅ Training completed successfully!")

if __name__ == '__main__':
    main()