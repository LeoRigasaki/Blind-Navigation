import onnxruntime as ort
import cv2
import numpy as np
import time

# Try to use CUDA if available
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
model = ort.InferenceSession("yolow-l.onnx", providers=providers)

class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "rickshaw", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "dog", "cat", "crosswalk", "curb", "pole", "street light", "trash can", "barrier", "sidewalk", "vendor cart", "bicycle stand", "parked car", "pothole", "speed breaker", "street vendor", "shop sign", "construction site", "open manhole", "water puddle", "billboard", "electrical box", "fence", "gate", "rickshaw"]

cap = cv2.VideoCapture("Walking Around New York City - Manhattan Videowalk【4K】🇺🇸.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_skip = 5  # Process every 5th frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    start_time = time.time()
    
    input_tensor = cv2.resize(frame, (640, 640)).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})
    
    print("\nDetected objects:")
    for box, score, class_id in zip(outputs[1][0], outputs[2][0], outputs[3][0]):
        if score > 0.05:
            x1, y1, x2, y2 = map(int, box * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_names[int(class_id)]}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            print(f"{label} at location ({x1}, {y1}, {x2}, {y2})")
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Video with Predictions", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()