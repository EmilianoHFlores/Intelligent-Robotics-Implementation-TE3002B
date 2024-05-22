# Takes n snapshots every s seconds

# Takes snapshots from a video and receives the x,y points form the user, then uses it them to calibrate the camera

import cv2
import numpy as np
import os
import math

# Create a VideoCapture object
video = cv2.VideoCapture("ducks.mp4")

# Takes n snapshots separated by s seconds
n = 5
s = 3
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = math.ceil(video.get(cv2.CAP_PROP_FPS))

# Create a directory to store the snapshots
if not os.path.exists("calibration"):
    os.makedirs("calibration")

FOLDER = f"calibration/calibration_{len(os.listdir('calibration'))}"
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
        
cv2.namedWindow("frame")
count = 0
# Loop through the video, when the frame is a snapshot, wait for the user to input the x,y points. A left click triggers the x,y terminal input, a right click triggers the next snapshot
for i in range(n_frames):
    # Read the frame
    ret, frame = video.read()
    
    # If the frame is not read, break the loop
    if not ret:
        break
    if i % (s * fps) == 0:
        cv2.imwrite(f"{FOLDER}/{i}.png", frame)
        count += 1
        if count == n:
            break