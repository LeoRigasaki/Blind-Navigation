import math
import pyttsx3
from ultralytics import YOLO
import cv2
import cvzone
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speaking speed
engine.setProperty('volume', 1)  # Set volume level

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width
cap.set(4, 640)   # Set the height

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Use a lighter model if available

# List of class names
classNames = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "rickshaw", "traffic light", "fire hydrant", 
    "stop sign", "parking meter", "bench", "dog", "cat", "crosswalk", "curb", "pole", "street light", 
    "trash can", "barrier", "sidewalk", "vendor cart", "bicycle stand", "parked car", "pothole", "speed breaker", 
    "street vendor", "shop sign", "construction site", "open manhole", "water puddle", "billboard", 
    "electrical box", "fence", "gate", "rickshaw"
]


# Limit the frame rate
prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to read frame from webcam.")
        break

    # Limit the frame rate to around 10 FPS
    new_frame_time = time.time()
    if new_frame_time - prev_frame_time < 0.1:
        continue
    prev_frame_time = new_frame_time

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            # Draw the bounding box with a corner rectangle
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Determine object location relative to the image center
            center_x = (x1 + x2) // 2
            img_width = img.shape[1]
            location = "on left, move right" if center_x < img_width // 2 else "on right, move left"
            cls_name = f'{class_name}'

            output_str = f'{class_name} detected {location}'

            # Display the class name
            cvzone.putTextRect(img, cls_name, (max(0, x1), max(35, y1)))

            # Speak out the class name and location (limit frequency)
            if conf > 0.5:  # Only speak if confidence is above 0.5
                engine.say(output_str)
                engine.runAndWait()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()