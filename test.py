import time
import numpy as np
import cv2
import signal
import sys
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model and allocate tensors.
def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load labels from a text file.
def load_labels(labels_path):
    with open(labels_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

# Get label index based on the desired label (e.g., "car" or "person")
def get_label_index(labels, target_label):
    if target_label in labels:
        return labels.index(target_label)
    else:
        raise ValueError(f"Label '{target_label}' not found in labels list.")

# Draw bounding boxes and labels on the frame
def draw_detections(frame, boxes, classes, scores, labels, threshold=0.5):
    for i in range(len(scores)):
        if scores[i] > threshold:
            class_id = int(classes[i])
            label = labels[class_id]
            box = boxes[i]
            y_min, x_min, y_max, x_max = box

            # Scale box coordinates back to frame size
            h, w, _ = frame.shape
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Draw label and score
            label_text = f"{label}: {scores[i]:.2f}"
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Run inference on a frame and check if the target object is detected.
def detect_objects(interpreter, frame, target_label, labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize frame to match model input shape
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    resized_frame = cv2.resize(frame, (input_width, input_height))

    # Convert the frame to RGB if it is not already in RGB format
    if len(resized_frame.shape) == 2:  # Grayscale
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
    elif resized_frame.shape[2] == 4:  # RGBA
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGBA2RGB)
    
    # Ensure the frame has the right shape and type
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    # Set the tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    detection_boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    detection_classes = interpreter.get_tensor(output_details[1]['index'])[0]
    detection_scores = interpreter.get_tensor(output_details[2]['index'])[0]

    draw_detections(frame, detection_boxes, detection_classes, detection_scores, labels)

    target_label_index = get_label_index(labels, target_label)
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5 and int(detection_classes[i]) == target_label_index:
            return True
    return False

# Signal handler to handle SIGINT and SIGTERM
def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)

# Main function
def main():
    model_path = "ssd_mobilenet_v1_coco_quant_postprocess.tflite"  # Update with your model path
    labels_path = "coco_labels.txt"  # Update with your labels file path
    labels = load_labels(labels_path)
    target_label = "car"  # Change this to the desired target label, e.g., "person"
    
    interpreter = load_model(model_path)

    encoder = H264Encoder(10000000)
    video_output = FfmpegOutput("video.mp4")

    picam2 = Picamera2()
    preview = Preview.QT

    # Handle signals for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    picam2.start_preview(preview)

    # Start recording with the output argument specified
    picam2.start_recording(encoder, output=video_output)
    start_time = time.time()
    
    try:
        while True:
            frame = picam2.capture_array()
            if detect_objects(interpreter, frame, target_label, labels):
                print(f"{target_label.capitalize()} detected! Saving 60-second clip.")
                picam2.stop_recording()
                time.sleep(60)  # Wait for 60 seconds
                picam2.start_recording(encoder, output=video_output)
                start_time = time.time()
            
            if time.time() - start_time >= 60:
                print("Saving next 60-second clip.")
                picam2.stop_recording()
                picam2.start_recording(encoder, output=video_output)
                start_time = time.time()
    except KeyboardInterrupt:
        print('Interrupted! Exiting...')
    finally:
        picam2.stop_recording()
        picam2.stop_preview()
        sys.exit(0)

if __name__ == '__main__':
    main()
