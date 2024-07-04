import onnx
import onnxruntime as ort
import cv2
import numpy as np
import time
import threading

# Load the ONNX model
onnx_model_path = "yolow-l.onnx"
session = ort.InferenceSession(onnx_model_path)

# Define the class names as per your model
class_names = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "rickshaw", "traffic light", "fire hydrant", 
    "stop sign", "parking meter", "bench", "dog", "cat", "crosswalk", "curb", "pole", "street light", 
    "trash can", "barrier", "sidewalk", "vendor cart", "bicycle stand", "parked car", "pothole", "speed breaker", 
    "street vendor", "shop sign", "construction site", "open manhole", "water puddle", "billboard", 
    "electrical box", "fence", "gate", "rickshaw"
]

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

def preprocess(image):
    # Preprocess the image for the model (resize and normalize)
    img = cv2.resize(image, (640, 640))
    img = img.transpose((2, 0, 1)).astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, original_shape):
    # Postprocess the model output to get bounding boxes, scores, and class IDs
    boxes = outputs[1][0]
    scores = outputs[2][0]
    class_ids = outputs[3][0]
    
    final_boxes, final_scores, final_class_ids = [], [], []
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > 0.05:  # Confidence threshold
            x1, y1, x2, y2 = box
            final_boxes.append([x1 * original_shape[1] / 640, y1 * original_shape[0] / 640,
                                x2 * original_shape[1] / 640, y2 * original_shape[0] / 640])
            final_scores.append(score)
            final_class_ids.append(int(class_id))
    
    return final_boxes, final_scores, final_class_ids

def draw_predictions(frame, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

# Define a thread-safe frame holder
class FrameHolder:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

frame_holder = FrameHolder()
processing_interval = 10  # Process every 10th frame

def capture_frames():
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % processing_interval == 0:
            frame_holder.set_frame(frame)

def process_frames():
    while cap.isOpened():
        frame = frame_holder.get_frame()
        if frame is None:
            continue
        original_shape = frame.shape
        input_tensor = preprocess(frame)
        outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
        boxes, scores, class_ids = postprocess(outputs, original_shape)
        frame = draw_predictions(frame, boxes, scores, class_ids)
        frame_holder.set_frame(frame)
        
        # Print detected objects
        print("\nDetected objects:")
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            print(f"{class_names[class_id]}: {score:.2f} at location ({x1}, {y1}, {x2}, {y2})")


# Start frame capture and processing threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

while cap.isOpened():
    frame = frame_holder.get_frame()
    if frame is not None:
        cv2.imshow("Webcam with Predictions", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_thread.join()
process_thread.join()

cap.release()
cv2.destroyAllWindows()
