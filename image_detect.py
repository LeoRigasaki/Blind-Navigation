import cv2
import numpy as np
import os

def process_image(image_path):
    # Read the image from the specified path
    frame = cv2.imread(image_path)
    
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

def process_images_in_folder(folder_path):
    # Get a list of all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(folder_path, image_file)
        
        # Process the image
        processed_image = process_image(image_path)
        
        # Display the processed image
        cv2.imshow('Sidewalk Detector', processed_image)
        
        # Wait for a key press and close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Path to the folder containing images
    folder_path = 'images'
    
    # Process all images in the folder
    process_images_in_folder(folder_path)

if __name__ == "__main__":
    main()
