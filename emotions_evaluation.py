from ultralytics import YOLO

model = YOLO('runs/detect/yolov11_custom_training3/weights/best.pt')

results = model.predict('image.jpeg',save=True,conf=0.35)