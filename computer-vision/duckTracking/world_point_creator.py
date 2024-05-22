# Relates the 2d points to the 3d points and saves them in a file

import cv2
import numpy as np
import os
import argparse

# first argument is the path to the image
parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to the image")
args = parser.parse_args()

# Load the image
image_file = args.image
frame = cv2.imread(image_file)
stop = False
image_x = -1
image_y = -1

def mouse_callback(event, x, y, flags, param):
    global image_x, image_y, stop
    if event == cv2.EVENT_LBUTTONDOWN:
        image_x = x
        image_y = y
    elif event == cv2.EVENT_RBUTTONDOWN:
        stop = True

# Create a window
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)
cv2.imshow("image", frame)

image_2d_points = []	
image_3d_points = []

while stop == False:
    for image_2d_point, image_3d_point in zip(image_2d_points, image_3d_points):
        frame = cv2.circle(frame, (image_2d_point[0], image_2d_point[1]), 5, (0, 0, 255), -1)
        frame = cv2.putText(frame, f"{image_3d_point[0]}, {image_3d_point[1]}", (image_2d_point[0], image_2d_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if image_x != -1 and image_y != -1:
        print(f"Image x: {image_x}, Image y: {image_y}")
        image_2d_points.append([image_x, image_y])
        input_x = input("Enter the x coordinate of the point in the world: ")
        input_y = input("Enter the y coordinate of the point in the world: ")
        print(f"World x: {input_x}, World y: {input_y}")
        image_3d_points.append([input_x, input_y])
        image_x = -1
        image_y = -1
        frame = cv2.circle(frame, (image_x, image_y), 5, (0, 0, 255), -1)
        frame = cv2.putText(frame, f"{input_x}, {input_y}", (image_x, image_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imshow("image", frame)
    cv2.waitKey(1)

# Save the points
print("image file: ", image_file)
image_file_name = os.path.splitext(image_file)[0]
print("image file name: ", image_file_name)
if os.path.exists(f"{image_file_name}_2d_points.npy"):
    os.remove(f"{image_file_name}_2d_points.npy")
if os.path.exists(f"{image_file_name}_3d_points.npy"):
    os.remove(f"{image_file_name}_3d_points.npy")
np.save(f"{image_file_name}_2d_points.npy", image_2d_points)
np.save(f"{image_file_name}_3d_points.npy", image_3d_points)
cv2.destroyAllWindows()
        