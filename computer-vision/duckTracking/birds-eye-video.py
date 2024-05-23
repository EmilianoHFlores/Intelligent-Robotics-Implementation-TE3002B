# Takes snapshots from a video and receives the x,y points form the user, then uses it them to calibrate the camera

import cv2
import numpy as np
import os
import math
import argparse


# Create a VideoCapture objec
POINTS_2D_FILE = "calibration/calibration_2/img0_2d_points.npy"
POINTS_3D_FILE = "calibration/calibration_2/img0_3d_points.npy"

points_2d = np.load(POINTS_2D_FILE)
points_3d = np.load(POINTS_3D_FILE)

def generate_camera_calibration(points_2d, points_3d):
    points_2d = np.array(points_2d, dtype=np.float32)

    # generate camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([points_3d], [points_2d], (640, 480), None, None)

    K = mtx
    D = dist

    print("Camera matrix : \n")
    print(K)

    focal_length = K[0][0]
    center = (K[0][2], K[1][2])

    print("Focal Length: ", focal_length)
    print("Center: ", center)

    return mtx, dist, rvecs, tvecs

def generate_birds_eye_homography(points_2d, points_3d, width, height):
    # get the homography
    h, status = cv2.findHomography(points_2d, points_3d)
    # modify so that we see more of the image
    # Move axis along x
    # apply the homography
    rotate = np.array([[-math.cos(math.pi/2), math.sin(math.pi/2), 0], [-math.sin(math.pi/2), -math.cos(math.pi/2), 0], [0, 0, 1]])
    h = np.dot(rotate, h)
    scaled_x = width / 2
    scaled_y = height / 2
    moveup = np.array([[1, 0, scaled_x/2], [0, 1, scaled_y], [0, 0, 1]])
    h = np.dot(moveup, h)
    multiplier = 2
    scaling = np.array([[multiplier, 0, 0], [0, multiplier, 0], [0, 0, 1]])
    # move in the green axis
    h = np.matmul(scaling, h)
    return h



SOURCE = "stabilized_video.avi"
cap = cv2.VideoCapture(SOURCE)


# camera calibration
mtx, dist, rvecs, tvecs = generate_camera_calibration(points_2d, points_3d)
# birds-eye homography
# get width, height from video
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
h = generate_birds_eye_homography(points_2d, points_3d, width, height)

while True:
    ret, frame = cap.read()
    birds_eye_frame = cv2.warpPerspective(frame, h, (width, height))
    cv2.imshow("Frame", frame)
    cv2.imshow("Birds Eye", birds_eye_frame)
    if not ret:
        break
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("s"):
        break

cv2.destroyAllWindows()
cap.release()