import onnxruntime as ort
import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
import cvzone
import pyttsx3

print("Initializing the application...")

# Load the ONNX model
onnx_model_path = "yolow-l.onnx"
print(f"Loading ONNX model from {onnx_model_path}...")
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
print("ONNX model loaded successfully.")

# Define the class names as per your model
class_names = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "rickshaw", "traffic light", "fire hydrant", 
    "stop sign", "parking meter", "bench", "dog", "cat", "crosswalk", "curb", "pole", "street light", 
    "trash can", "barrier", "sidewalk", "vendor cart", "bicycle stand", "parked car", "pothole", "speed breaker", 
    "street vendor", "shop sign", "construction site", "open manhole", "water puddle", "billboard", 
    "electrical box", "fence", "gate", "rickshaw"
]

# Initialize text-to-speech engine
print("Initializing text-to-speech engine...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate
print("Text-to-speech engine initialized.")

def preprocess(image):
    img = cv2.resize(image, (640, 640))  # Reduced input size for faster processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, original_shape):
    boxes = outputs[1][0]
    scores = outputs[2][0]
    class_ids = outputs[3][0]
    
    detections = [(box, score, class_id) for box, score, class_id in zip(boxes, scores, class_ids) if score > 0.3]
    detections.sort(key=lambda x: x[1], reverse=True)
    detections = detections[:5]
    
    final_boxes, final_scores, final_class_ids = [], [], []
    
    for box, score, class_id in detections:
        x1, y1, x2, y2 = box
        final_boxes.append([int(x1 * original_shape[1] / 320), int(y1 * original_shape[0] / 320),
                            int(x2 * original_shape[1] / 320), int(y2 * original_shape[0] / 320)])
        final_scores.append(score)
        final_class_ids.append(int(class_id))
    
    return final_boxes, final_scores, final_class_ids

def draw_predictions(frame, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Draw bounding box
        cvzone.cornerRect(frame, (x1, y1, w, h))
        
        # Prepare label
        label = f'{class_names[class_id]} {score:.2f}'
        
        # Draw label
        cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1)), 
                           scale=1, thickness=1, colorR=(0,255,0))
    return frame

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

def inference_thread(input_queue, output_queue):
    print("Inference thread started.")
    while True:
        frame = input_queue.get()
        if frame is None:
            break
        input_tensor = preprocess(frame)
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        boxes, scores, class_ids = postprocess(outputs, frame.shape)
        output_queue.put((boxes, scores, class_ids))
    print("Inference thread stopped.")

def audio_thread(audio_queue):
    print("Audio thread started.")
    while True:
        message = audio_queue.get()
        if message is None:
            break
        engine.say(message)
        engine.runAndWait()
    print("Audio thread stopped.")

def generate_audio_guidance(boxes, class_ids, frame_shape, audio_queue):
    height, width = frame_shape[:2]
    center_x = width // 2
    
    critical_objects = []
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2
        
        if obj_center_y > height * 0.5:  # Object in lower half of frame
            if obj_center_x < center_x:
                direction = "left"
            else:
                direction = "right"
            
            critical_objects.append((class_names[class_id], direction))
    
    if critical_objects:
        message = "Warning! "
        for obj, direction in critical_objects:
            message += f"{obj} on your {direction}. "
        audio_queue.put(message)

# Initialize video capture
print("Initializing video capture...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Video capture initialized.")

# Create queues and start threads
input_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
audio_queue = Queue()

print("Starting inference thread...")
inference_thread = Thread(target=inference_thread, args=(input_queue, output_queue))
inference_thread.start()

print("Starting audio thread...")
audio_thread = Thread(target=audio_thread, args=(audio_queue,))
audio_thread.start()

prev_frame_time = 0
audio_cooldown = 0

print("Starting main loop...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    input_tensor = preprocess(frame)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    boxes, scores, class_ids = postprocess(outputs, frame.shape)
    
    # Generate audio guidance
    if audio_cooldown == 0:
        generate_audio_guidance(boxes, class_ids, frame.shape, audio_queue)
        audio_cooldown = 5  # Set cooldown to avoid constant audio updates
    else:
        audio_cooldown -= 1
    
    # Draw predictions and boundary
    frame = draw_predictions(frame, boxes, scores, class_ids)
    frame = draw_leaning_boundary(frame)
    
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time) if current_time - prev_frame_time > 1e-6 else 0
    prev_frame_time = current_time
    cvzone.putTextRect(frame, f"FPS: {fps:.2f}", (10, 30), 
                       scale=3, thickness=3, colorR=(0,255,0))
    
    cv2.imshow("Blind Navigation Assistant", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User requested to quit. Exiting...")
        break

print("Cleaning up...")
# Signal the threads to stop
input_queue.put(None)
audio_queue.put(None)

print("Waiting for threads to finish...")
inference_thread.join()
audio_thread.join()

cap.release()
cv2.destroyAllWindows()
print("Application terminated.")
