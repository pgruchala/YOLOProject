from ultralytics import YOLO


model = YOLO('yolo11n.pt')
results = model('piotrus.jpg', save=True, show=True)