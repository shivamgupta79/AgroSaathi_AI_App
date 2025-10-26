from ultralytics import YOLO

model = YOLO("models/best.pt")

# Export to ONNX
model.export(format="onnx")

# Export to TorchScript
model.export(format="torchscript")
