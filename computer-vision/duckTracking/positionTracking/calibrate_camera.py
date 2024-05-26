# Takes snapshots from a video and receives the x,y points form the user, then uses it them to calibrate the camera

import cv2
import numpy as np
import os
import math

# Create a VideoCapture objec
POINTS_2D_FILE = "calibration/calibration_2/img0_2d_points.npy"
POINTS_3D_FILE = "calibration/calibration_2/img0_3d_points.npy"

points_2d = np.load(POINTS_2D_FILE)
points_3d = np.load(POINTS_3D_FILE)


print("Points 2D and 3D: ")

print(points_2d)
print(points_3d)
print(f"Number of points 2d: {len(points_2d)}")
print(f"Number of points 3d: {len(points_3d)}")

image = cv2.imread("calibration/calibration_2/img0.png")

# draw each 2d point andw rite what 3d point it is
for i in range(len(points_3d)):
    cv2.circle(image, tuple(points_2d[i]), 5, (0, 0, 255), -1)
    cv2.putText(image, f"{points_3d[i]}", tuple(points_2d[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

# birds eye view
# create a 3d axis
axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)

# project 3d points to image plane
imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)

# draw the axis
image = cv2.imread("calibration/calibration_2/img0.png")
image = cv2.drawFrameAxes(image, mtx, dist, rvecs[0], tvecs[0], 10)

# warp so that the blue axis is 0, meaning we see it perm¿pendicular to it
# get the homography
# get the homography
h, status = cv2.findHomography(points_2d, points_3d)
# modify so that we see more of the image
# Move axis along x
# apply the homography
rotate = np.array([[-math.cos(math.pi/2), math.sin(math.pi/2), 0], [-math.sin(math.pi/2), -math.cos(math.pi/2), 0], [0, 0, 1]])
h = np.dot(rotate, h)
scaled_x = image.shape[1] / 2
scaled_y = image.shape[0] / 2
moveup = np.array([[1, 0, scaled_x/2], [0, 1, scaled_y], [0, 0, 1]])
h = np.dot(moveup, h)
multiplier = 2
scaling = np.array([[multiplier, 0, 0], [0, multiplier, 0], [0, 0, 1]])
# move in the green axis
h = np.matmul(scaling, h)
birds_eye = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))
# draw the axis


cv2.imshow("Image", image)

cv2.imshow("Birds Eye", birds_eye)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()