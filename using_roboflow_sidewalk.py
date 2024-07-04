import cv2
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="YGdi5nzniOtNQ57mZxYo")
project = rf.workspace().project("blind-naviagtion")
roboflow_model = project.version(1).model

def draw_arrow(frame, start_point, end_point, color, thickness=2):
    """Draw an arrow on the frame pointing from start_point to end_point."""
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.5)

def process_frame(frame):
    # Run Roboflow model inference
    results = roboflow_model.predict(frame, confidence=40).json()['predictions']
    
    sidewalks = []
    obstacles = []

    for result in results:
        x, y, w, h = result['x'], result['y'], result['width'], result['height']
        if result['class'] == 'sidewalk':
            sidewalks.append((x, y, w, h))
        elif result['class'] == 'obstacle':
            obstacles.append((x, y, w, h))
    
    nearest_sidewalk = None
    min_distance = float('inf')
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    for (x, y, w, h) in sidewalks:
        # Calculate boundary points
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        
        # Draw bounding box
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
        # Calculate and mark midpoint
        midpoint = (int(x), int(y))
        cv2.circle(frame, midpoint, 5, (0, 0, 255), -1)
        
        # Calculate distances to boundaries
        distance_left = midpoint[0]
        distance_right = frame.shape[1] - midpoint[0]
        distance_top = midpoint[1]
        distance_bottom = frame.shape[0] - midpoint[1]
        
        # Display distances
        cv2.putText(frame, f'Left: {distance_left}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Right: {distance_right}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Top: {distance_top}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Bottom: {distance_bottom}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        

        # Determine direction
        center_x = x
        direction = "Move Forward"
        if center_x < frame.shape[1] // 3:
            direction = "Move Left"
        elif center_x > 2 * frame.shape[1] // 3:
            direction = "Move Right"
        
        # Display direction
        cv2.putText(frame, direction, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calculate distance from frame center to sidewalk midpoint
        distance = np.sqrt((frame_center[0] - midpoint[0]) ** 2 + (frame_center[1] - midpoint[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_sidewalk = midpoint
    
    # Draw arrow pointing to the nearest sidewalk if no obstacle on sidewalk
    if nearest_sidewalk:
        draw_arrow(frame, frame_center, nearest_sidewalk, (0, 0, 255), 3)

    for (x, y, w, h) in obstacles:
        # Calculate boundary points
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        
        # Draw bounding box
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        
        # Display label
        cv2.putText(frame, "Obstacle", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Check if obstacle is on a sidewalk
        for (sx, sy, sw, sh) in sidewalks:
            sidewalk_top_left = (int(sx - sw / 2), int(sy - sh / 2))
            sidewalk_bottom_right = (int(sx + sw / 2), int(sy + sh / 2))
            if sidewalk_top_left[0] <= x <= sidewalk_bottom_right[0] and sidewalk_top_left[1] <= y <= sidewalk_bottom_right[1]:
                draw_arrow(frame, (x, y), (sx, sy), (0, 255, 0), 3)
                break
    
    return frame

def process_video(video_path, output_path=None, skip_frames=5):
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
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        # if frame_count % skip_frames == 0:
        #     processed_frame = process_frame(frame)
        # else:
        #     processed_frame = frame
        processed_frame = process_frame(frame)

        # Display the frame
        cv2.imshow('Sidewalk and Obstacle Detector', processed_frame)
        
        # Write the frame to the output video if specified
        if output_path:
            out.write(processed_frame)
        
        frame_count += 1
        
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
    video_path = 'walking-around-new-york-city-manhattan-videowalk-4k.mp4'
    
    # Optional: Path to save the processed video
    output_path = 'manhattan_video.avi'
    
    # Process the video
    process_video(video_path, output_path, skip_frames=0)

if __name__ == "__main__":
    main()
