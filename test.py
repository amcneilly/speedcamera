import cv2
import time
import numpy as np
import os
import argparse
import threading
from picamera2 import Picamera2, Preview
from libcamera import controls
from tflite_runtime.interpreter import Interpreter

# def periodic_autofocus(picam2, interval=4):
#     while True:
#         picam2.set_controls({"AfMode": 1 ,"AfTrigger": 0})
#         time.sleep(interval)

# Argument parser setup
parser = argparse.ArgumentParser(description='Object detection and recording script.')
parser.add_argument('--desired_object', type=str, default='car', help='Object to detect')
parser.add_argument('--recording_duration', type=int, default=15, help='Recording duration in seconds')
parser.add_argument('--video_resolution', type=str, default='800x600', help='Desired video resolution (widthxheight)')
parser.add_argument('--output_folder', type=str, default='recordings', help='Folder to save videos')

args = parser.parse_args()

# Configuration
model_path = "ssd_mobilenet_v1_coco_quant_postprocess.tflite"  # Update with your model path
labels_path = "coco_labels.txt"  # Update with your labels file path
desired_object = args.desired_object  # Object to detect
recording_duration = args.recording_duration  # Recording duration in seconds
video_resolution = tuple(map(int, args.video_resolution.split('x')))  # Desired video resolution (width, height)
model_input_size = (300, 300)  # Model input size (width, height)
vid_fps = 20  # Video frames per second
zoom_value = 1.0  # Zoom value
output_folder = args.output_folder  # Folder to save videos

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

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
picam2.start_preview(Preview.QTGL)
print("video_resolution = " + str(video_resolution))
config = picam2.create_preview_configuration(main={"size": video_resolution})
picam2.configure(config)
picam2.start()

#set auto focus
print("Applying autofocus")
time.sleep(1)
picam2.set_controls({"AfMode": 2 ,"AfTrigger": 0})
time.sleep(5)

# Start the periodic autofocus thread
# autofocus_thread = threading.Thread(target=periodic_autofocus, args=(picam2,))
# autofocus_thread.daemon = True  # Daemonize the thread to ensure it exits when the main program does
# autofocus_thread.start()

# Apply zoom
#picam2.set_controls({"Zoom": zoom_value})

def detect_objects(frame):
    input_data = cv2.resize(frame, model_input_size)  # Resize to model input size
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
    detection_made = False
    for i in range(len(boxes)):
        if scores[i] > threshold:
            class_id = int(classes[i])
            label = labels.get(class_id, "Unknown")
            if label == desired_object:
                detection_made = True
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                ymin = int(ymin * height)
                xmin = int(xmin * width)
                ymax = int(ymax * height)
                xmax = int(xmax * width)
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detection_made

def add_timestamp(frame):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def microcontroller_on_detection():
    print("Microcontroller: Object detected")

def microcontroller_on_recording_start():
    print("Microcontroller: Recording started")

def microcontroller_on_recording_end():
    print("Microcontroller: Recording ended")

recording = False
recording_end_time = 0
detection_flag = False

while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for detection
    boxes, classes, scores = detect_objects(frame_rgb)
    frame, detection_made = draw_boxes(frame, boxes, classes, scores)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert frame back to BGR for display
    
    if detection_made and not recording and not detection_flag:
        print(f"Detection made at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        microcontroller_on_detection()
        detection_flag = True  # Set the flag to indicate detection

    if recording:
        frame_bgr = add_timestamp(frame_bgr)  # Add timestamp to frame
        video_writer.write(frame_bgr)
        if time.time() > recording_end_time:
            print(f"Recording ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            recording = False
            video_writer.release()
            microcontroller_on_recording_end()
            detection_flag = False  # Reset the flag after recording ends
    
    if not recording and detection_flag:
        print(f"Recording started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        recording = True
        recording_end_time = time.time() + recording_duration
        filename = os.path.join(output_folder, time.strftime("%Y%m%d_%H%M%S") + ".avi")
        video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), vid_fps, video_resolution)
        microcontroller_on_recording_start()

    #cv2.imshow("Object Detection", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
