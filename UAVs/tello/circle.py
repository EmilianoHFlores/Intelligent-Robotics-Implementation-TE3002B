# Makes a circle with the drone
from djitellopy import Tello
import math
import time
from threading import Thread
import datetime
import cv2
import numpy as np

class CircleDrone:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        #self.tello.set_speed(25) # 25cm/s
        self.recording = False

    def takeoff(self):
        self.tello.takeoff()
    
    def land(self):
        self.tello.land()

    def make_circle(self):
        SPEED = 50 # applies to both x and y, so 12.5cm/s if 50%
        
        CIRCLE_RADIUS = 100 # cm
        
        speed_x = 0
        speed_y = 0
        angle = 0

        while angle < 360:
            speed_x = SPEED * math.cos(math.radians(angle))
            speed_y = SPEED * math.sin(math.radians(angle))
            
            self.tello.send_rc_control(int(speed_x), int(speed_y), 0, 0)
            time.sleep(0.025)
            angle += 1
    
    def start_recording(self):
        self.recording = True
        thread = Thread(target=self.record)
        thread.start()
    
    def wait_for_stream(self):
        while not self.stream_started:
            pass
        return
    
    def stop_recording(self):
        self.recording = False
    
    def record(self):
        self.stream_started = False
        FILENAME = f"circle_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.avi"
        print(f"Recording video as {FILENAME}")
        self.tello.streamon()
        tello_video = self.tello.get_frame_read()
        height, width, _ = tello_video.frame.shape
        print("Trying to get frame")
        image_mean = 0
        while image_mean < 10: # check image is not all black
            image_mean = np.mean(tello_video.frame)
        print(f"Got frame: {height} {width}")
        self.stream_started = True
        video = cv2.VideoWriter(FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))
        while self.recording:
            video.write(tello_video.frame)
            print(tello_video.frame)
            time.sleep(1/30)
        
        video.release()
        print(f"Video saved as {FILENAME}")
        

# Usage:
circle_drone = CircleDrone()
print("Taking off")
circle_drone.start_recording()
circle_drone.wait_for_stream()
#circle_drone.takeoff()
print("Making circle")
#circle_drone.make_circle()
#circle_drone.land()
time.sleep(25)
circle_drone.stop_recording()




