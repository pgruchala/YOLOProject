import cv2
import torch
from ultralytics import YOLO
import time

model = YOLO('emotions140.pt') #runs/detect/yolov11_custom_training3/weights/best.pt

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
        
    results = model(frame)
    
    annotated_frame = results[0].plot()
    
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv11 Webcam", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()