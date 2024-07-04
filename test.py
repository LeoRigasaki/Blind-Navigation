import cv2
import numpy as np

def draw_leaning_boundary(frame, color=(0, 255, 0), thickness=2):
    height, width = frame.shape[:2]
    
    # Define the corners of the leaning rectangle
    top_left = (int(width * 0.2), int(height * 0.3))
    top_right = (int(width * 0.8), int(height * 0.3))
    bottom_left = (int(width * 0.1), int(height * 0.9))
    bottom_right = (int(width * 0.9), int(height * 0.9))
    
    # Draw the leaning rectangle
    cv2.line(frame, top_left, top_right, color, thickness)
    cv2.line(frame, bottom_left, bottom_right, color, thickness)
    cv2.line(frame, top_left, bottom_left, color, thickness)
    cv2.line(frame, top_right, bottom_right, color, thickness)
    
    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw the leaning boundary
    frame_with_boundary = draw_leaning_boundary(frame)
    
    # Display the result
    cv2.imshow('Webcam with Leaning Boundary', frame_with_boundary)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()