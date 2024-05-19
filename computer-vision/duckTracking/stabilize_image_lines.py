import numpy as np  
import time
import cv2
import imutils
# cv2.namedWindow('matches', 0)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
# cv2.namedWindow('motion', cv2.WINDOW_NORMAL)

FRAME_HEIGHT = 720

class OrbPoints:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, img):
        orb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb_image = cv2.resize(orb_image, (640, 360))
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def match_descriptors(self, des1, des2):
        try:
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
        except:
            matches = []
        return matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        try:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        except:
            img3 = img1
        return img3
    
    def calculate_motion(self, kp1, kp2, matches):
        if len(matches) <= 0:
            return np.zeros((100, 100, 3)), 0, np.zeros(2)

        # get motion from one frame to another  
        print("--------------------")   
        matched_points_1 = np.array([kp1[match.queryIdx].pt for match in matches])
        matched_points_2 = np.array([kp2[match.trainIdx].pt for match in matches])
        # print(f"matched_points_1: {matched_points_1}")
        # print(f"matched_points_2: {matched_points_2}")
        # draw each in a different plot, match the points with the same index with color
        # fig = plt.figure(figsize=(15,15))
        # plt.tight_layout()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224)
        # Assigning colors to each point
        colors = np.random.rand(len(matched_points_1), 3)
        # Plotting the first set of points
        # ax1.scatter(matched_points_1[:, 0], matched_points_1[:, 1], c=colors)
        # ax1.set_title('First set of points')
        # Plotting the second set of points
        # ax2.scatter(matched_points_2[:, 0], matched_points_2[:, 1], c=colors)
        # ax2.set_title('Second set of points')
        
        center_1 = np.mean(matched_points_1, axis=0)
        center_2 = np.mean(matched_points_2, axis=0)
        
        # Plotting the centers of the points
        # ax1.scatter(center_1[0], center_1[1], c='r', marker='x')
        # ax2.scatter(center_2[0], center_2[1], c='r', marker='x')
        
        # center each set of points at the origin
        centered_points_1 = matched_points_1 - center_1
        centered_points_2 = matched_points_2 - center_2
        
        # plot the centered points
        # ax3 = fig.add_subplot(223)
        # ax3.scatter(centered_points_1[:, 0], centered_points_1[:, 1], c=colors)
        # ax3.scatter(centered_points_2[:, 0], centered_points_2[:, 1], c=colors)
        # ax3.scatter(0,0, c='r', marker='x')
        
        # calculate the rotation matrix
        H = np.dot(centered_points_1.T, centered_points_2)
        
        # calculate the singular value decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # calculate the rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # calculate the translation vector
        t = center_2 - np.dot(R, center_1)
        
        # calculate the transformed points
        transformed_points = np.dot(centered_points_1, R.T) + t
        
        # plot the transformed points
        # ax4 = fig.add_subplot(224)
        # ax4.scatter(transformed_points[:, 0], transformed_points[:, 1], c=colors)
        # ax4.scatter(centered_points_2[:, 0], centered_points_2[:, 1], c=colors)
        
        # plot the center of the points
        # ax4.scatter(0,0, c='r', marker='x')
        
        # angle of rotation
        angle = np.arctan2(R[1, 0], R[0, 0])
        # x translation
        t[0]
        # y translation
        t[1]
        
        print(f"angle: {angle}")
        print(f"t: {t}")
        print("--------------------")
        
        fig = np.zeros((100, 100, 3))
        return fig, angle, t


def get_lines(frame):
    # filter to detect lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # slight erode
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    # slight dilate
    kernel = np.ones((1, 1), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
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

def get_vertical_lines(line_list):
    vertical_lines = []
    # draw vertical lines
    for theta, rho in line_list:
        m = -1 / np.tan(theta)
        b = rho / np.sin(theta)
        # filter close to vertical lines
        if abs(m) < 3:
            continue
        vertical_lines.append([theta, rho])
    return vertical_lines

def get_most_vertical_line(line_list):
    # get the most vertical line
    max_m = 0
    max_line = None
    print("get_most_vertical_line")
    for theta, rho in line_list:
        m = -1 / np.tan(theta)
        if abs(m) > max_m:
            max_m = abs(m)
            max_line = [theta, rho]
    return max_line

def get_lines_angle(line_list):
    # get average angle from vertical lines
    first_analysis = False
    positive_m = False
    
    sum_m = 0
    count = 0
    for theta, rho in line_list:
        m = -1 / np.tan(theta)
        b = rho / np.sin(theta)
        # filter close to vertical lines
        if not first_analysis:
            if m > 0:
                first_analysis = True
                positive_m = True
        else:
            if positive_m and m < 0:
                continue
            if not positive_m and m > 0:
                continue
        sum_m += m
        count += 1
    if count == 0:
        return 0
    angle = np.arctan(sum_m / count)
    return angle

def get_line_angle(line):
    m = -1 / np.tan(line[0])
    return np.arctan(m)
        

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

def show_only_lines(frame, line_list):
    # obscure everything but the lines
    binary_image = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    for theta, rho in line_list:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(binary_image, (x1, y1), (x2, y2), 255, 2)
    # dilate binary
    # print(binary_image.shape)
    # kernel = np.ones((5, 5), np.uint8)
    # binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    # binary_image = np.reshape(binary_image, (binary_image.shape[0], binary_image.shape[1], 1))
    return frame & binary_image

        
orb_points = OrbPoints()
video = cv2.VideoCapture("ducks.mp4")

ret, prev_frame = video.read()
sample_frame = imutils.resize(prev_frame, height=FRAME_HEIGHT)
# image is cut to only show as much as the height, so when it is rotated, no black bars are shown
# cv2.resizeWindow('matches', sample_frame.shape[0], sample_frame.shape[0])
# cv2.resizeWindow('motion', sample_frame.shape[0], sample_frame.shape[0])
# cv2.resizeWindow('motion', sample_frame.shape[1], sample_frame.shape[0])

black_frame = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), np.uint8)

angle_accum = 0
angle = 0
rotate_angle = 0

while ret:
    ret, frame = video.read()
    if not ret:
        break

    prev_frame_analysis = prev_frame.copy()
    # apply the stabilization to the frame
    # cut the frame to only show the height, so instead of width, height, it is height, height
    cut_length = (prev_frame_analysis.shape[1] - prev_frame_analysis.shape[0]) // 2 + 30
    
    prev_frame_analysis = prev_frame_analysis[:, cut_length:cut_length + prev_frame_analysis.shape[0]]
    cv2.imshow("prev_frame_raw", prev_frame_analysis)
    
    prev_frame_analysis = imutils.rotate(prev_frame_analysis, rotate_angle)
    print("prev_frame_analysis.shape", prev_frame_analysis.shape)
    print("cut_length", cut_length)
    
    cv2.imshow("prev_frame", prev_frame_analysis)
    line_list = get_lines(prev_frame_analysis)
    # prev_frame_analysis = draw_lines(black_frame, line_list)
    prev_frame_lines = show_only_lines(prev_frame_analysis, line_list)
    
    vertical_list = get_vertical_lines(line_list)
    if len(vertical_list) == 0 and len(line_list) > 0:
        vertical_list.append(get_most_vertical_line(line_list))
    else:
        print("vertical_list", vertical_list)
    most_vertical = get_most_vertical_line(vertical_list)
    angle = get_line_angle(most_vertical) * 180 / np.pi + 90
    if angle > 90:
        angle = angle - 180
    rotate_angle += angle
    prev_frame_vertical_lines = draw_lines(prev_frame_analysis, vertical_list)
    cv2.imshow("prev_frame_vertical_lines", prev_frame_vertical_lines)
    print("Rotate angle: ", rotate_angle)
    # frame_analysis = frame.copy()
    # frame_analysis = frame_analysis[:frame_analysis.shape[0], :frame_analysis.shape[0]]
    # line_list = get_lines(frame_analysis)Q
    # #frame_analysis = draw_lines(black_frame, line_list)
    # frame_analysis = show_only_lines(frame_analysis, line_list)
    # kp1, des1 = orb_points.detect_and_compute(prev_frame_analysis)
    # kp2, des2 = orb_points.detect_and_compute(frame_analysis)

    # matches = orb_points.match_descriptors(des1, des2)
    
    # fig, angle, translation = orb_points.calculate_motion(kp1, kp2, matches)
    
    
    
    # stabilzie angle
    # angle_accum += angle
    # print("angle_accum", angle_accum)
    
    rotated_frame = imutils.rotate(frame, angle)  
    # fig.canvas.draw()
    # print("transformnig to array")
    # img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    # img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGRA2RGB)
    # print(img_plot.shape)

    #img3 = orb_points.draw_matches(prev_frame, kp1, frame, kp2, matches)
    #print("drawn matches")
    
    #cv2.imshow("motion", img_plot)
    # cv2.imshow("matches", img3)
    # cv2.imshow("motion", img_plot)
    cv2.imshow("rotated", rotated_frame)
    cv2.imshow("lines", prev_frame_lines)
    
    # wait until a key is pressed to continue
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
    prev_frame = frame

cv2.destroyAllWindows()
video.release()