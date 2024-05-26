import cv2
import numpy as np

# takes a snapshot from the nth video frame
def take_snapshot(video_path, frame_number, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Set the frame position to the desired frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = video.read()
    video.release()
    # Check if the frame was read successfully
    if ret:
        # Save the frame as an image
        return True, frame
    else:
        return False, np.array([])

# Example usage
video_path = "ducks.mp4"
frame_number = 100
output_path = "ducks.jpg"
take_snapshot(video_path, frame_number, output_path)