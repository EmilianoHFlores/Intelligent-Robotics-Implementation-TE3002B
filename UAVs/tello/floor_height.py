import djitellopy
import time
# Connect to Tello
tello = djitellopy.Tello()
tello.connect()

# Get the current height of the drone
while True:
    print(tello.get_distance_tof())
    time.sleep(1)

