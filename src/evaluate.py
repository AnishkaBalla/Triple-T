from ultralytics import YOLO

# 1. Load trained model
model = YOLO(r"C:\Users\anish\Triple-T\yolov8n.pt")

# 2. Evaluate on validation dataset
metrics = model.val(data=r"C:\Users\anish\Triple-T\data.yaml", plots=True, conf=0.25, iou=0.6)

mAP50_95 = metrics.box.map       # mAP@50-95
mAP50 = metrics.box.map50        # mAP@50
precision = metrics.box.p        # Array of precision per class
recall = metrics.box.r           # Array of recall per class

print(f"mAP@50: {mAP50}")
print(f"mAP@50-95: {mAP50_95}")