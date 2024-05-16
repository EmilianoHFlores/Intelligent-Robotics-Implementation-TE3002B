# Using https://github.com/SAmmarAbbas/birds-eye-view we obtained camera intrinsic parameters
# Output of the code
# ------------------------------------------------
# Horizon Line: [-1.91608007e-04  2.36453646e-03  1.00000000e+00]
# Vertical Vanishing Point: [  23.57359628 5234.87806116]
# fx: 1762.3818658905311 fy: 1771.882532753107 roll: 4.632787333400636 tilt(rad): 0.3398831745192811 tilt(deg): 19.47387142746321
# Focal Length of the camera (pixels): 1771.882532753107
# Roll of the camera (degrees): 4.632787333400636
# Tilt of the camera (degrees): 19.47387142746321
# [-9139 -3061]
# [-9139 -3061]
# target dim:  (1460, 2160)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from take_snapshot_from_video import take_snapshot

VIDEO_FILE = "ducks.mp4"
video = cv2.VideoCapture(VIDEO_FILE)
width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Camera intrinsic parameters
fx = 1762.3818658905311
fy = 1771.882532753107
x0 = width / 2
y0 = height / 2

K = np.array([[fx, 0, x0],
                [0, fy, y0],
                [0, 0, 1]])

# Camera extrinsic parameters will be obtained from a vanishing point
result, frame = take_snapshot(VIDEO_FILE, 40, "ducks.jpg")

def get_lines(frame):
    # filter to detect lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # slight erode
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    # slight dilate
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 250)
    line_list = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # discard similar lines
            if len(line_list) > 0:
                skip = False
                for line in line_list:
                    if abs(line[0] - theta) < 100 and abs(line[1] - rho) < 35:
                        skip = True
                        break
                if skip:
                    continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            line_list.append([theta, rho])
    
    return line_list

def draw_lines(frame, line_list):
    analysis_image = frame.copy()
    for theta, rho in line_list:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(analysis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return analysis_image

def get_py_from_vp(u_i, v_i, K):
    p_infinity = np.array([u_i, v_i, 1])
    K_inv = np.linalg.inv(K)
    r3 = K_inv @ p_infinity    
    r3 /= np.linalg.norm(r3)
    yaw = -np.arctan2(r3[0], r3[2])
    pitch = np.arcsin(r3[1])    
    
    return pitch, yaw

if result:
    # Vanishing point will be obtained from the grid lines on the floor
    analysis_image = frame.copy()
    # filter to detect lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # slight erode
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    # slight dilate
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 250)
    line_list = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            # discard similar lines
            if len(line_list) > 0:
                skip = False
                for line in line_list:
                    if abs(line[0] - theta) < 100 and abs(line[1] - rho) < 35:
                        skip = True
                        break
                if skip:
                    continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(analysis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_list.append([theta, rho])
    # subplots
    print(line_list)
    fig, axs = plt.subplots(2, 3)
    # increase size
    fig.set_size_inches(20, 10)
    # tight layout
    plt.tight_layout()
    axs[0][0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0][0].set_title("Original Image")
    axs[0][1].imshow(edges, cmap='gray')
    axs[0][1].set_title("Canny")
    axs[0][2].imshow(cv2.cvtColor(analysis_image, cv2.COLOR_BGR2RGB))
    axs[0][2].set_title("Detected Lines")
    
    
    # calculate vanishing point
    # filter only semivertical lines (m is around pi/2), range is variable
    vanishing_image = frame.copy()
    vertical_lines = []
    # draw vertical lines
    for theta, rho in line_list:
        m = -1 / np.tan(theta)
        b = rho / np.sin(theta)
        # filter close to vertical lines
        if abs(m) < 0.8:
            continue
        x0 = 0
        y0 = int(b)
        x1 = -1000
        y1 = int(m * x1 + b)
        x2 = 1000
        y2 = int(m * x2 + b)
        vertical_lines.append([m, b])
        # cv2.line(vanishing_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    # calculate vanishing point
    vanishing_point_x_accum = 0
    vanishing_point_y_accum = 0
    vanishing_points = []
    for m, b in vertical_lines:
        for m2, b2 in vertical_lines:
            if m == m2:
                continue
            A = np.array([[m, -1], [m2, -1]])
            B = np.array([-b, -b2])
            vanishing_point = np.linalg.solve(A, B)
            vanishing_point = vanishing_point.astype(int)
            vanishing_points.append([vanishing_point[0], vanishing_point[1]])
            vanishing_point_x_accum += vanishing_point[0]
            vanishing_point_y_accum += vanishing_point[1]
    num_combinations = len(vertical_lines) * (len(vertical_lines) - 1)
    vanishing_point = [vanishing_point_x_accum / num_combinations, vanishing_point_y_accum / num_combinations]
    # calculate intersection of first 2 lines
    # m1, b1 = vertical_lines[0]
    # m2, b2 = vertical_lines[1]
    # A = np.array([[m1, -1], [m2, -1]])
    # B = np.array([-b1, -b2])
    # vanishing_point = np.linalg.solve(A, B)
    # vanishing_point = vanishing_point.astype(int)
    
    
    print("Vanishing Point: ", vanishing_point)
    axs[1][0].imshow(cv2.cvtColor(vanishing_image, cv2.COLOR_BGR2RGB))
    # draw lines
    for m, b in vertical_lines:
        if vanishing_point[0] < 0:
            x1 = vanishing_point[0] - 100
            y1 = int(m * x1 + b)
            x2 = vanishing_image.shape[1]
            y2 = int(m * x2 + b)
        else:
            x1 = 0
            y1 = int(b)
            x2 = vanishing_image.shape[1] + 100
            y2 = int(m * x2 + b) 
        axs[1][0].plot([x1, x2], [y1, y2], c='b')
    axs[1][0].scatter(vanishing_point[0], vanishing_point[1], c='r', s=100, marker='o')
    print(vanishing_points)
    # ax2.scatter([point[0] for point in vanishing_points], [point[1] for point in vanishing_points], c='r', s=100)
    axs[1][0].set_title("Vanishing Point")
    
    pitch, yaw = get_py_from_vp(vanishing_point[0], vanishing_point[1], K)
    
    
    plt.show()
    
    
    
    
else:
    print("Failed to take snapshot")
    
