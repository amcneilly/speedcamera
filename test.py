import cv2
import time
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# Configuration
model_path = "ssd_mobilenet_v1_coco_quant_postprocess.tflite"  # Update with your model path
labels_path = "coco_labels.txt"  # Update with your labels file path
desired_object = "car"  # Object to detect
recording_duration = 30  # Recording duration in seconds

# Load labels
with open(labels_path, 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Load TensorFlow Lite model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

def detect_objects(frame):
    input_data = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    input_data = np.uint8(input_data)  # Convert input data to uint8
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    
    return boxes, classes, scores

def draw_boxes(frame, boxes, classes, scores, threshold=0.5):
    height, width, _ = frame.shape
    for i in range(len(boxes)):
        if scores[i] > threshold:
            class_id = int(classes[i])
            label = labels.get(class_id, "Unknown")
            if label == desired_object:
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * height)
                xmin = int(xmin * width)
                ymax = int(ymax * height)
                xmax = int(xmax * width)
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

recording = False
recording_end_time = 0

while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for detection
    boxes, classes, scores = detect_objects(frame_rgb)
    frame = draw_boxes(frame, boxes, classes, scores)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert frame back to BGR for display
    
    cv2.imshow("Object Detection", frame_bgr)
    
    if not recording:
        for i in range(len(classes)):
            if labels.get(int(classes[i]), "") == desired_object and scores[i] > 0.5:
                recording = True
                recording_end_time = time.time() + recording_duration
                filename = time.strftime("%Y%m%d_%H%M%S") + ".avi"
                video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame.shape[1], frame.shape[0]))
                break

    if recording:
        video_writer.write(frame_bgr)
        if time.time() > recording_end_time:
            recording = False
            video_writer.release()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
