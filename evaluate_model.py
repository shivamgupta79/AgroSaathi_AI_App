from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

# Evaluate on dataset
results = model.val(data="dataset/data.yaml", save_json=True)
print("Results:", results)
