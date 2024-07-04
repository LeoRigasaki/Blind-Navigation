import cv2
import numpy as np

# Load the MobileNet-SSD model
prototxt_path = 'deploy.prototxt'
model_path = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Class labels for the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Track previous positions of detected people
previous_positions = {}

def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append((startX, startY, endX, endY, CLASSES[idx]))
    return results

def predict_direction(current_pos, prev_pos):
    # Predict the movement direction based on previous and current positions
    if current_pos[0] > prev_pos[0]:
        return "right"
    elif current_pos[0] < prev_pos[0]:
        return "left"
    else:
        return "straight"

def process_frame(frame):
    global previous_positions
    height, width, _ = frame.shape
    center_x = width // 2
    
    # Detect objects
    detections = detect_objects(frame)
    
    # Initialize path points (straight line down the middle)
    path_points = [(center_x, height), (center_x, height // 2)]
    
    # Determine the safest path by avoiding detected objects
    for (startX, startY, endX, endY, label) in detections:
        if label in ["person", "bicycle", "car", "bus", "motorbike"]:
            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Predict movement direction
            current_pos = ((startX + endX) // 2, (startY + endY) // 2)
            if label not in previous_positions:
                previous_positions[label] = current_pos
                direction = "straight"
            else:
                direction = predict_direction(current_pos, previous_positions[label])
                previous_positions[label] = current_pos

            # Adjust the path to avoid the object
            if direction == "right":
                path_points[0] = (center_x - width // 4, height)  # Move path to the left
                path_points[1] = (center_x - width // 4, height // 2)
            elif direction == "left":
                path_points[0] = (center_x + width // 4, height)  # Move path to the right
                path_points[1] = (center_x + width // 4, height // 2)
            elif startX < center_x < endX:
                path_points[0] = (center_x - width // 4, height)  # Move path to the left
                path_points[1] = (center_x - width // 4, height // 2)
            elif startX < center_x + width // 4 and endX > center_x:
                path_points[0] = (center_x + width // 4, height)  # Move path to the right
                path_points[1] = (center_x + width // 4, height // 2)

    # Draw the path line
    for i in range(len(path_points) - 1):
        cv2.arrowedLine(frame, path_points[i], path_points[i + 1], (0, 255, 0), 3, tipLength=0.5)
    
    # Draw direction text
    if path_points[0][0] < center_x:
        direction_text = "Move Left"
    elif path_points[0][0] > center_x:
        direction_text = "Move Right"
    else:
        direction_text = "Move Forward"
    
    cv2.putText(frame, direction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_path:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display the frame
        cv2.imshow('Path Planning', processed_frame)
        
        if output_path:
            out.write(processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    video_path = 'PersonCollision.mp4'
    output_path = 'processed_video.avi'
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()
