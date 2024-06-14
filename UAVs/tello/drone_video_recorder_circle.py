from djitellopy import Tello
import math
import time
import cv2
from threading import Thread
import os

VIDEO_FOLDER = "circle_videos"

if not os.path.exists(VIDEO_FOLDER):
    os.mkdir(VIDEO_FOLDER)
tello = Tello()

print("Connecting to Tello")
tello.connect()

tello.streamon()

frame_reader = tello.get_frame_read()

keepRecording = True

def thread_func():
    # Main loop to display the video stream
    VIDEO_FILE = os.path.join(VIDEO_FOLDER, f"Circle_{time.strftime('%Y%m%d-%H%M%S')}.avi")
    recorder = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*'XVID'), 60, (960, 720))
    try:
        while keepRecording:
            frame = frame_reader.frame
            filter = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Tello Camera Filtered", filter)
            recorder.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        recorder.release()
        print(f"Recording stored at {VIDEO_FILE}")

recorder = Thread(target=thread_func)
recorder.start()

# print("Taking off")
# tello.takeoff()

# time.sleep(1)

# period = 30
# radius = 0.7

# linear_velocity = 2 * math.pi * radius / period
# time_delta = period / 360

# for i in range(360):
#     linear_velocity_x = linear_velocity * math.cos(math.radians(i))
#     linear_velocity_y = linear_velocity * math.sin(math.radians(i))
#     x = int(100 * (linear_velocity_x/linear_velocity) / 6)
#     y = int(100 * (linear_velocity_y/linear_velocity) / 6)
#     tello.send_rc_control(x, y, 0, 0)
#     time.sleep(time_delta)

#     print (f"Linear velocity x: {x}, Linear velocity y: {y}, Degree: {i}")

# print("Stopping")
# tello.send_rc_control(0, 0, 0, 0)

time.sleep(30)

# print("Landing")
# tello.land()

keepRecording = False
recorder.join()
print("All stopped")