from ultralytics import YOLO

model = YOLO('runs/detect/yolov11_custom_training3/weights/best.pt')

results = model.predict(source=0,save=True,conf=0.25)