import cv2
import time
import numpy as np
import os
import argparse
import threading
from picamera2 import Picamera2, Preview
from libcamera import controls

def periodic_autofocus(picam2, AFMode=1, aftrigger=0, interval=30):
    print("Periodic autofocus started with interval =", interval)
    while True:
        picam2.set_controls({"AfMode": AFMode, "AfTrigger": aftrigger})
        time.sleep(interval)

# Argument parser setup
parser = argparse.ArgumentParser(description='Object detection and recording script.')
parser.add_argument('--recording_duration', type=int, default=15, help='Recording duration in seconds')
parser.add_argument('--video_resolution', type=str, default='1024x760', help='Desired video resolution (widthxheight)')
parser.add_argument('--output_folder', type=str, default='recordings', help='Folder to save videos')
parser.add_argument('--preview', action='store_true', help='Show preview')
parser.add_argument('--AFMode', type=int, default=1, help='Auto Focus Mode')
parser.add_argument('--aftrigger', type=int, default=0, help='Auto Focus Trigger')
parser.add_argument('--exposure', type=int, help='Set camera exposure time')
parser.add_argument('--interval', type=int, default=30, help='Interval for periodic autofocus in seconds')
parser.add_argument('--fps', type=int, default=30, help='Frames per second for video recording')

args = parser.parse_args()

# Configuration
video_resolution = args.video_resolution  # Desired video resolution (widthxheight)
video_width, video_height = tuple(map(int, video_resolution.split('x')))
vid_fps = args.fps  # Frames per second for video recording
output_folder = args.output_folder  # Folder to save videos

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
if args.preview:
    picam2.start_preview(Preview.QTGL)
print("video_resolution = " + str(video_resolution))
config = picam2.create_preview_configuration(main={"size": (video_width, video_height), "format": "YUV420"})
picam2.configure(config)
picam2.start()

# Set exposure time if provided
if args.exposure:
    print(f"Setting exposure to {args.exposure}")
    picam2.set_controls({"ExposureTime": args.exposure})

# Start the periodic autofocus thread
autofocus_thread = threading.Thread(target=periodic_autofocus, args=(picam2, args.AFMode, args.aftrigger, args.interval))
autofocus_thread.daemon = True  # Daemonize the thread to ensure it exits when the main program does
autofocus_thread.start()

recording = False
recording_end_time = 0
video_writer = None
frame_count = 0

# Start recording immediately for the duration specified
recording = True
recording_end_time = time.time() + args.recording_duration
filename = os.path.join(output_folder, time.strftime("%Y%m%d_%H%M%S") + ".mp4")
video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (video_width, video_height))
print(f"Initialized VideoWriter with filename={filename}, fourcc=mp4v, fps={vid_fps}, resolution=({video_width}, {video_height})")

while recording:
    frame = picam2.capture_array()
    frame_count += 1

    # Ensure frame is in BGR format
    if len(frame.shape) == 2:  # If the frame is grayscale, convert it to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 1:  # If frame has only one channel, convert to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    video_writer.write(frame)
    print(f"Writing frame {frame_count} to video file.")

    if time.time() > recording_end_time:
        print(f"Recording ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        recording = False

if video_writer is not None:
    video_writer.release()
    print("Released VideoWriter.")

picam2.stop()
cv2.destroyAllWindows()
