import cv2
import numpy as np
import os

# Configuration
video_width, video_height = 1024, 760
vid_fps = 30
output_folder = 'recordings'
filename = os.path.join(output_folder, "test_static.mp4")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize VideoWriter
video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), vid_fps, (video_width, video_height))
print(f"Initialized VideoWriter with filename={filename}, fourcc=mp4v, fps={vid_fps}, resolution=({video_width}, {video_height})")

# Generate static frames
for i in range(100):  # 100 frames for approx 3 seconds at 30 FPS
    frame = np.random.randint(0, 256, (video_height, video_width, 3), dtype=np.uint8)
    video_writer.write(frame)
    print(f"Writing static frame {i+1} to video file.")

if video_writer is not None:
    video_writer.release()
    print("Released VideoWriter.")

# Verify video file size
file_size = os.path.getsize(filename)
print(f"Generated video file size: {file_size} bytes")
