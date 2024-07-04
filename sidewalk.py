from roboflow import Roboflow
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="YGdi5nzniOtNQ57mZxYo")
project = rf.workspace().project("blind-naviagtion")
roboflow_model = project.version(1).model

# Initialize YOLOv8n
yolo_model = YOLO('yolov5nu.pt')

# Open the video file
video_path = "Walking Around New York City - Manhattan Videowalk【4K】🇺🇸.mp4"
cap = cv2.VideoCapture(0)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process every 5th frame to reduce computation while maintaining smooth playback
    if frame_count % 5 == 0:
        # Perform inference on the frame using Roboflow model (sidewalk detection)
        roboflow_prediction = roboflow_model.predict(frame, confidence=20, overlap=30).json()
        
        # Perform inference using YOLOv8n
        yolo_results = yolo_model(frame)
        
        # Draw Roboflow predictions (sidewalks)
        for prediction in roboflow_prediction['predictions']:
            x = int(prediction['x'])
            y = int(prediction['y'])
            w = int(prediction['width'])
            h = int(prediction['height'])
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(frame, f"Sidewalk: {prediction['confidence']:.2f}", (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw YOLOv8n predictions
        for result in yolo_results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = result.names[cls]
                if label in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'traffic light', 'stop sign']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Print the prediction results
        print(f"Frame {frame_count} prediction:")
        print("Sidewalks detected:", len(roboflow_prediction['predictions']))
        print("YOLO detections:", len(boxes))
    
    # Display the frame
    cv2.imshow('Video with Detections', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames.")