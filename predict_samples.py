import glob
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

for img_path in glob.glob("dataset/images/val/*.jpg"):
    results = model(img_path)
    annotated = results[0].plot()
    cv2.imshow("Prediction", annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
