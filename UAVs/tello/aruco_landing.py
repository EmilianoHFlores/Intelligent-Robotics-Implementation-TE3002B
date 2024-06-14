from djitellopy import Tello
import math
import time
import cv2
from threading import Thread
import os
import numpy as np

camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

VIDEO_FOLDER = "final_videos"

if not os.path.exists(VIDEO_FOLDER):
    os.mkdir(VIDEO_FOLDER)
tello = Tello()

print("Connecting to Tello")
tello.connect()

tello.streamon()

frame_reader = tello.get_frame_read()

keepRecording = True

aruco_pos_x = 999
aruco_distance = -1

def thread_func():
    # Main loop to display the video stream
    #VIDEO_FILE = os.path.join(VIDEO_FOLDER, f"Circle_{time.strftime('%Y%m%d-%H%M%S')}.avi")
    #recorder = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*'XVID'), 60, (960, 720))
    global aruco_pos_x, aruco_distance
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    MARKER_SIZE = 20.0 # cm
    UNDETECTED_FRAME_LIMIT = 150
    undetected_count = 0
    try:
        while keepRecording:
            frame = frame_reader.frame
            filter = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            
            if ids is not None:
                undetected_count = 0
                for i in range(len(ids)):
                    if ids[i]==2:
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, camera_matrix, distortion)
                        cv2.aruco.drawDetectedMarkers(filter, corners) 
                        cv2.drawFrameAxes(frame, camera_matrix, distortion, rvec, tvec, 0.1) 
                        # print("tvec", tvec)
                        aruco_pos_x = tvec[0][0][0]
                        aruco_distance = tvec[0][0][2]
                        cv2.putText(filter, f"Distance: {aruco_distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                undetected_count+=1
                if undetected_count>UNDETECTED_FRAME_LIMIT:
                    aruco_pos_x = 999
            cv2.imshow("Tello Camera Filtered", filter)
            #recorder.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()
        # recorder.release()
        # print(f"Recording stored at {VIDEO_FILE}")

camera = Thread(target=thread_func)
camera.start()

tello.takeoff()
tello.send_rc_control(0,0,0,0)

# while not keyboard interrupt
yaw_speed = 0
TOP_YAW_SPEED = 50
p = 0.5
x_pos_tolerance  = 10

# requires n seconds with the aruco in the center to land
required_stabilized_time = 5
stabilized = False
ready_to_land = False

# samples of height and distance to consider
number_of_samples = 10
 
try:
    while True:
        print("Distance:", aruco_distance)
        if abs(aruco_pos_x) > x_pos_tolerance and not ready_to_land:
            stabilized = False
            yaw_speed = aruco_pos_x*p
            if aruco_pos_x>0:
                print("Rotating -")
            else:
                print("Rotating +")
            print("Sending yaw: ", yaw_speed)
            if abs(yaw_speed) > TOP_YAW_SPEED:
                yaw_speed = TOP_YAW_SPEED * abs(yaw_speed)/yaw_speed
            tello.send_rc_control(0,0,0,int(yaw_speed))
            
        else:
            tello.send_rc_control(0,0,0,0)
            if not stabilized:
                stabilized_time = time.time()
                stabilized = True
            if time.time() - stabilized_time > required_stabilized_time:
                print("Ready to land")
                ready_to_land = True
            print("Not Moving")
        print("Stabilized: ", stabilized)
        print("Ready to land: ", ready_to_land)
        if ready_to_land:
            distance_accum = 0
            height_accum = 0
            for i in range(number_of_samples):
                distance_accum+=aruco_distance
                height_accum+=tello.get_distance_tof()
            land_distance = distance_accum/number_of_samples
            drone_height = height_accum/number_of_samples
            print(f"Aruco is at distance: {land_distance}, drone is at height: {drone_height}")
            drone_distance_from_aruco = math.sqrt(land_distance**2-drone_height**2)
            print(f"Drone is at {drone_distance_from_aruco} cm from the aruco horizontally")
            drone_distance_from_aruco = math.floor(drone_distance_from_aruco)
            while drone_distance_from_aruco>500:
                tello.move_forward(500)
                drone_distance_from_aruco-=500
            tello.move_forward(drone_distance_from_aruco)
                
            break
                
        print("-"*20)
        time.sleep(0.1)
        pass
except KeyboardInterrupt:
    pass

tello.send_rc_control(0,0,0,0)
# print("Landing")
tello.land()

keepRecording = False
camera.join()
print("All stopped")