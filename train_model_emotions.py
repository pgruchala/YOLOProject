from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(data='emotions_dataset/data.yaml',
                      epochs=140,      
                      imgsz=96,      
                      batch=16,        
                      name='yolov11_emotions'
                      )