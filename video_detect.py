import cv2
import numpy as np

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use color segmentation to detect the sidewalk (assuming it's greyish)
    lower_bound = np.array([90, 90, 90], dtype="uint8")
    upper_bound = np.array([200, 200, 200], dtype="uint8")
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    
    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the sidewalk
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Determine direction
        center_x = x + w // 2
        direction = "Move Forward"
        if center_x < frame.shape[1] // 3:
            direction = "Move Left"
        elif center_x > 2 * frame.shape[1] // 3:
            direction = "Move Right"
        
        # Draw bounding box and direction
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def process_video(video_path, output_path=None):
    # Capture video from the file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a VideoWriter object if output path is specified
    if output_path:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display the frame
        cv2.imshow('Sidewalk Detector', processed_frame)
        
        # Write the frame to the output video if specified
        if output_path:
            out.write(processed_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    # Path to the video file
    video_path = 'PersonCollision.mp4'
    
    # Optional: Path to save the processed video
    output_path = 'processed_video.avi'
    
    # Process the video
    process_video(video_path, output_path)

if __name__ == "__main__":
    main()
