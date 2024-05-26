import numpy as np  
import time
import cv2
import math
import pickle
import imutils
cv2.namedWindow('Analysis', 0)
import matplotlib
matplotlib.use('TkAgg') 

from matplotlib import pyplot as plt
cv2.namedWindow('Analysis', cv2.WINDOW_NORMAL)
import argparse 

SOURCE = "stabilized_video_ducks.avi"
TRACKS_SOURCE = "frame_tracks.pkl"
CALIBRATION_2D_POINTS_FILE = "calibration/calibration_2/img0_2d_points.npy"
CALIBRATION_3D_POINTS_FILE = "calibration/calibration_2/img0_3d_points.npy"

class CameramanPositionTracker:
    def __init__(self, source, calibration_2d_points_file, calibration_3d_points_file, save_video=''):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        self.frame_tracks = pickle.load(open(TRACKS_SOURCE, "rb"))
        
        # self.frame_tracks[0] is a dict that ties id with the bounding boxes
        self.track_num = len(self.frame_tracks[0])
        print("Number of tracks: ", self.track_num)
        
        self.ducks_positions = {duck_id : [] for duck_id in self.frame_tracks[0].keys()}
        self.ducks_current_x = {duck_id : 0 for duck_id in self.frame_tracks[0].keys()}
        self.ducks_current_y = {duck_id : 0 for duck_id in self.frame_tracks[0].keys()}
        # create a color for each track
        self.colors = np.random.randint(0, 255, (self.track_num, 3))
        # assign to each id
        self.track_colors = {}
        for i, track_id in enumerate(self.frame_tracks[0].keys()):
            self.track_colors[track_id] = self.colors[i]
        
        self.FIG_SIZE_X = 30
        self.FIG_SIZE_Y = 25
        
        self.FRAME_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FRAME_WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.save_video = save_video
        if save_video != '':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(save_video, fourcc, self.FPS, (self.FIG_SIZE_X*100, self.FIG_SIZE_Y*100))
        # ORB
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.calibration_2d_points = np.load(calibration_2d_points_file)
        self.calibration_3d_points = np.load(calibration_3d_points_file)

        self.grid_filter = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), np.uint8)
        self.filter_alpha = 0.7
        grid_divisions = 10
        for i in range(grid_divisions):
            for j in range(grid_divisions):
                # generate according to rainbow spectrum
                hue = int(255 * (i * grid_divisions + j) / (grid_divisions**2 - 1))
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)
                self.grid_filter[i * (self.FRAME_HEIGHT // grid_divisions): (i + 1) * (self.FRAME_HEIGHT // grid_divisions), j * (self.FRAME_WIDTH // grid_divisions): (j + 1) * (self.FRAME_WIDTH // grid_divisions)] = color
        
        # instead of each pixel being 1cm, it will be multiplied by this value
        self.homography_visualization_multiplier = 2
        self.camera_matrix, self.distortion_coefficients, self.rotation_vectors, self.translation_vectors = self.generate_camera_calibration(self.calibration_2d_points, self.calibration_3d_points)
        self.homography = self.generate_birds_eye_homography(self.calibration_2d_points, self.calibration_3d_points, 640, 480)
        self.birds_eye_view = None
        
        # 4x2 matplotlib figure
        self.fig = plt.figure(figsize=(30,25))
        self.axs = [self.fig.add_subplot(2, 3, 1), self.fig.add_subplot(2, 3, 2), self.fig.add_subplot(2, 3, 3), self.fig.add_subplot(2, 3, 4), self.fig.add_subplot(2, 3, 5), self.fig.add_subplot(2, 3, 6)]
        # plt.tight_layout()
        self.fig.canvas.draw()
        
        
        self.axs[0].set_title("Original Image")
        self.axs[1].set_title("Birds Eye View")
        self.axs[2].set_title("Line Detection")
        self.axs[3].set_title("Orb Matches")
        self.axs[4].set_title("Cameraman Position")
        self.axs[5].set_title("Ducks Position")
        
        self.cameraman_angle = 0
        self.cameraman_positions = []
        self.cameraman_current_x = 0
        self.cameraman_current_y = 0
        
        # queues for moving average
        self.cameraman_positions_x_queue = []
        self.cameraman_positions_y_queue = []
        self.cameraman_angle_queue = []
        
        
        self.width = self.FRAME_WIDTH
        self.height = self.FRAME_HEIGHT
        
        self.orig_img_show = self.axs[0].imshow(np.zeros((self.height, self.width, 3)))
        self.birds_eye_show = self.axs[1].imshow(np.zeros((self.height, self.width, 3)))
        self.line_detection_show = self.axs[2].imshow(np.zeros((self.height, self.width, 3)))
        self.orb_show = self.axs[3].imshow(np.zeros((self.height, self.width, 3)))
        # cameraman position will be a line plot where the cameraman new positions will be added
        self.cameraman_position_plot = self.axs[4].plot([], [])[0]
        # The ducks position will be a plot with several lines, one for each duck
        # allows for several series
        self.ducks_positions_plot = self.axs[5]
        
        # use blit
        self.axs_background = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axs]
                
        self.analyze_video()
        
    def generate_camera_calibration(self, points_2d, points_3d):
        points_2d = np.array(points_2d, dtype=np.float32)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([points_3d], [points_2d], (640, 480), None, None)
        K = mtx
        D = dist
        focal_length = K[0][0]
        center = (K[0][2], K[1][2])
        return mtx, dist, rvecs, tvecs

    def generate_birds_eye_homography(self, points_2d, points_3d, width, height):
        h, status = cv2.findHomography(points_2d, points_3d)
        rotate = np.array([[-math.cos(math.pi/2), math.sin(math.pi/2), 0], [-math.sin(math.pi/2), -math.cos(math.pi/2), 0], [0, 0, 1]])
        h = np.dot(rotate, h)
        scaled_x = width / 2
        scaled_y = height / 2
        moveup = np.array([[1, 0, scaled_x/2], [0, 1, scaled_y], [0, 0, 1]])
        h = np.dot(moveup, h)
        multiplier = self.homography_visualization_multiplier
        scaling = np.array([[multiplier, 0, 0], [0, multiplier, 0], [0, 0, 1]])
        h = np.matmul(scaling, h)
        return h
    
    def plt_to_cv2(self, fig):
        
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGRA2RGB)
        
        return img_plot
    
    def draw(self, ax_n, img, title):
        show_img = self.axs[ax_n].set_data(img)
        # cv2.imshow(str(ax_n), img)
        self.axs[ax_n].set_title(title)
        self.axs[ax_n].draw_artist(show_img)
        
    def get_useful_area(self, img):
        # get the useful area of the image, which is that occupied by the transformed image
        # Biggest contour is used
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return mask
    
    def detect_lines(self, img):
        
        # filter to detect lines
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 100, 130, apertureSize=3)
        # slight erode
        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
        # slight dilate
        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
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
        
    def show_only_lines(self, frame, line_list):
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
            cv2.line(binary_image, (x1, y1), (x2, y2), 255, 4)
        # dilate binary
        # print(binary_image.shape)
        # kernel = np.ones((5, 5), np.uint8)
        # binary_image = cv2.dilate(binary_image, kernel, iterations=1)
        # binary_image = np.reshape(binary_image, (binary_image.shape[0], binary_image.shape[1], 1))
        return frame & binary_image
    
    def apply_blurred_filter(self, frame, filter_frame, alpha):
        blurred_filter = cv2.GaussianBlur(filter_frame, (21, 21), 0)
        print(frame.shape, blurred_filter.shape)
        return cv2.addWeighted(frame, alpha, blurred_filter, 1 - alpha, 0)
    
    def calculate_matching_points(self, prev_img, img, useful_area, prev_useful_area):
        
        def detect_and_compute(self, img):
            orb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            orb_image = cv2.resize(orb_image, (640, 360))
            kp, des = self.orb.detectAndCompute(img, None)
            return kp, des
                    
            return filtered_matches
        
        def match_descriptors(self, des1, des2):
            try:
                matches = self.bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
            except:
                matches = []
            return matches
        
        def filter_matches(self, kp1, kp2, matches, useful_area, prev_useful_area):
            # filter matches that are far from each other, since this is a video, they should be close
            matched_points_1_raw = np.array([kp1[match.queryIdx].pt for match in matches])
            matched_points_2_raw = np.array([kp2[match.trainIdx].pt for match in matches])
            filtered_matches = []
            # if distance between points is too large, it is likely an error
            # also if it is around the edges, it is the edge of the image
            
            # erode useful areas to get a wider margin for filtering
            kernel = np.ones((20,20), np.uint8)
            useful_area_filter = useful_area.copy()
            prev_useful_area_filter = prev_useful_area.copy()
            
            useful_area_filter = cv2.erode(useful_area, kernel, iterations=2)
            prev_useful_area_filter = cv2.erode(prev_useful_area, kernel, iterations=2)
            
            for i, (matched_point_1, matched_point_2) in enumerate(zip(matched_points_1_raw, matched_points_2_raw)):
                distance_between_points = np.linalg.norm(matched_point_1 - matched_point_2)
                # further filter, if it is around the edges, it is likely noise
                if distance_between_points > 10:
                    continue
                if not useful_area_filter[int(matched_point_1[1]), int(matched_point_1[0])]:
                    continue
                if not prev_useful_area_filter[int(matched_point_2[1]), int(matched_point_2[0])]:
                    continue
                filtered_matches.append(matches[i])
        
            return filtered_matches

        kp1, des1 = detect_and_compute(self, prev_img)
        kp2, des2 = detect_and_compute(self, img)
        
        matches = match_descriptors(self, des1, des2)
        filtered_matches = filter_matches(self, kp1, kp2, matches, useful_area, prev_useful_area)
        
        matches_image = cv2.drawMatches(prev_img, kp1, img, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        return kp1, kp2, filtered_matches, matches_image
    
    def calculate_motion(self, kp1, kp2, matches, width, height):
        if len(matches) <= 0:
            return 0, np.zeros(2)

        # get motion from one frame to another  
        matched_points_1 = np.array([kp1[match.queryIdx].pt for match in matches])
        matched_points_2 = np.array([kp2[match.trainIdx].pt for match in matches])  
        
        if len(matched_points_1) <= 0:
            return 0, np.zeros(2)
        colors = np.random.rand(len(matched_points_1), 3)
        center_1 = np.mean(matched_points_1, axis=0)
        center_2 = np.mean(matched_points_2, axis=0)
        
        # center each set of points at the origin
        centered_points_1 = matched_points_1 - center_1
        centered_points_2 = matched_points_2 - center_2
        
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
        
        # angle of rotation
        angle = np.arctan2(R[1, 0], R[0, 0])
        # x translation
        t[0]
        # y translation
        t[1]
        
        # print(f"angle: {angle}")
        # print(f"t: {t}")
        # print("--------------------")
        return angle, t
    
    def calculate_motion_essential(self, kp1, kp2, matches, width, height):
        if len(matches) <= 0:
            return 0, np.zeros(2)

        # get motion from one frame to another  
        matched_points_1 = np.array([kp1[match.queryIdx].pt for match in matches])
        matched_points_2 = np.array([kp2[match.trainIdx].pt for match in matches])  
        
        try:
            # calculate the essential matrix
            E, mask = cv2.findEssentialMat(matched_points_1, matched_points_2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # recover pose
            _, R, t, _ = cv2.recoverPose(E, matched_points_1, matched_points_2, self.camera_matrix)
        except:
            return 0, np.zeros(2)        
        # angle of rotation
        angle = np.arctan2(R[1, 0], R[0, 0])
        # x translation
        t[0]
        # y translation
        t[1]
        
        # print(f"angle: {angle}")
        # print(f"t: {t}")
        # print("--------------------")
        return angle, t
    
    def update_cameraman_position_plot(self):
        # plot the cameraman position
        self.cameraman_position_plot.set_xdata([pos[0] for pos in self.cameraman_positions])    
        self.cameraman_position_plot.set_ydata([pos[1] for pos in self.cameraman_positions])
    
    def get_duck_bboxes(self, frame_count):
        duck_bboxes = {}
        for i, frame_track_id in enumerate(self.frame_tracks[frame_count]):
            frame_track = self.frame_tracks[frame_count][frame_track_id]
            if len(frame_track) > 0:
                duck_bboxes[frame_track_id] = frame_track
        return duck_bboxes
    
    def get_duck_centroids(self, frame_count):
        duck_centroids = {}
        for i, frame_track_id in enumerate(self.frame_tracks[frame_count]):
            frame_track = self.frame_tracks[frame_count][frame_track_id]
            if len(frame_track) > 0:
                width = frame_track[2] - frame_track[0]
                height = frame_track[3] - frame_track[1]
                x = frame_track[0] + width // 2
                y = frame_track[1] + height // 2
                duck_centroids[frame_track_id] = (x, y)
        return duck_centroids
    
    def draw_duck_bboxes(self, frame, duck_bboxes):
        
        for duck_bbox_id in duck_bboxes:
            x1, y1, x2, y2 = duck_bboxes[duck_bbox_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.track_colors[duck_bbox_id].tolist(), 2)
        return frame
            
    def draw_duck_centroids(self, frame, duck_centroids):
        for duck_centroid_id in duck_centroids:
            x, y = duck_centroids[duck_centroid_id]
            cv2.circle(frame, (x, y), 5, self.track_colors[duck_centroid_id].tolist(), -1)
        return frame
            
    def warp_duck_bboxes(self, h, duck_bboxes, duck_centroids):
        warped_duck_bboxes = {}
        warped_duck_centroids = {}
        for duck_bbox_id in duck_bboxes:
            x1, y1, x2, y2 = duck_bboxes[duck_bbox_id]
            x, y = duck_centroids[duck_bbox_id]
            # warp the bounding box
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            warped_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), h)
            warped_points = warped_points.reshape(-1, 2)
            x1, y1 = warped_points[0]
            x2, y2 = warped_points[2]
            
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            
            # warp centroid to homography
            print("Centroid x, y: ", x, y)
            centroid = np.array([[x, y]], dtype=np.float32)
            warped_centroid = cv2.perspectiveTransform(centroid.reshape(-1, 1, 2), h)
            warped_centroid = warped_centroid.reshape(-1, 2)
            x, y = warped_centroid[0]
            x, y = int(x), int(y)
            print("Warped centroid x, y: ", x, y)
            
            warped_duck_bboxes[duck_bbox_id] = (x1, y1, x2, y2)
            warped_duck_centroids[duck_bbox_id] = (x, y)
        return warped_duck_bboxes, warped_duck_centroids
        
    def calculate_ducks_motion(self, prev_warped_centroids, warped_centroids):
        # generate a dictionaries with 0 for every duck_id
        ducks_x_motion = {duck_id : 0 for duck_id in self.ducks_positions}
        ducks_y_motion = {duck_id : 0 for duck_id in self.ducks_positions}
        print("Prev warped centroids: ", prev_warped_centroids)
        print("Warped centroids: ", warped_centroids)
        for duck_id in ducks_x_motion:
            try:
                x, y = prev_warped_centroids[duck_id]
                x_new, y_new = warped_centroids[duck_id]
                x_motion = x_new - x
                y_motion = y_new - y
                print("Motion for duck: ", duck_id, x_motion, y_motion)
                ducks_x_motion[duck_id] = x_motion
                ducks_y_motion[duck_id] = y_motion
            except:
                print("Motion for duck (untracked): ", duck_id, 0, 0)
                ducks_x_motion[duck_id] = 0
                ducks_y_motion[duck_id] = 0
        return ducks_x_motion, ducks_y_motion
            
    def analyze_video(self):
        
        # plt.show(block=False)   
        frame_count = 0
        ret, prev_frame = self.cap.read()
        prev_duck_bboxes = self.get_duck_bboxes(frame_count)
        prev_duck_centroids = self.get_duck_centroids(frame_count)
        prev_warped_bboxes, prev_warped_centroids = self.warp_duck_bboxes(self.homography, prev_duck_bboxes, prev_duck_centroids)
        prev_birds_eye_view = cv2.warpPerspective(prev_frame, self.homography, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        prev_useful_area = self.get_useful_area(prev_birds_eye_view)
        prev_line_list = self.detect_lines(prev_birds_eye_view)
        prev_line_image = self.show_only_lines(prev_birds_eye_view, prev_line_list)
        prev_line_image = self.apply_blurred_filter(prev_line_image, self.grid_filter, self.filter_alpha)
        
        frame_count+=1
        
        while True:
            
            # clean canvas
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            duck_bboxes = self.get_duck_bboxes(frame_count)
            duck_centroids = self.get_duck_centroids(frame_count)
            
            warped_bboxes, warped_centroids = self.warp_duck_bboxes(self.homography, duck_bboxes, duck_centroids)
            
            birds_eye_view = cv2.warpPerspective(frame, self.homography, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            useful_area = self.get_useful_area(birds_eye_view)
            
            line_list = self.detect_lines(birds_eye_view)
            line_image = self.show_only_lines(birds_eye_view, line_list)
            line_image = self.apply_blurred_filter(line_image, self.grid_filter, self.filter_alpha)
            line_image = cv2.bitwise_and(line_image, line_image, mask=prev_useful_area)
            
            # ORB analysis
            
            kp1, kp2, matches, matches_image = self.calculate_matching_points(prev_line_image, line_image, useful_area, prev_useful_area)
            
            
            angle, t = self.calculate_motion(kp1, kp2, matches, self.FRAME_WIDTH, self.FRAME_HEIGHT)
            
            t_0 = -t[0]
            t_1 = -t[1]
            
            # moving average
            NUMBER_SAMPLES = 10
            if len(self.cameraman_positions_x_queue) >= NUMBER_SAMPLES:
                self.cameraman_positions_x_queue.pop(0)
                self.cameraman_positions_y_queue.pop(0)
                self.cameraman_angle_queue.pop(0)
            self.cameraman_positions_x_queue.append(t_0)
            self.cameraman_positions_y_queue.append(t_1)
            self.cameraman_angle_queue.append(angle)
            
            print(f"t: {t}")
            
            t_0 = sum(self.cameraman_positions_x_queue) / len(self.cameraman_positions_x_queue)
            t_1 = sum(self.cameraman_positions_y_queue) / len(self.cameraman_positions_y_queue)
            angle = sum(self.cameraman_angle_queue) / len(self.cameraman_angle_queue)
            
            self.cameraman_angle += angle
            print(f"angle: {angle}")    
            x_motion = (t_0 / self.homography_visualization_multiplier) * math.cos(self.cameraman_angle)
            y_motion = (t_1 / self.homography_visualization_multiplier) * math.sin(self.cameraman_angle)
            self.cameraman_current_x += x_motion
            self.cameraman_current_y += y_motion
            self.cameraman_current_position = [self.cameraman_current_x, self.cameraman_current_y]
            self.cameraman_positions.append(self.cameraman_current_position)
            
            ducks_x_motion, ducks_y_motion = self.calculate_ducks_motion(prev_warped_centroids, warped_centroids)
            
            for duck_id in ducks_x_motion:
                if ducks_x_motion[duck_id] == 0 and ducks_y_motion[duck_id] == 0:
                    continue
                ducks_x_motion[duck_id] = ducks_x_motion[duck_id] / self.homography_visualization_multiplier - x_motion
                ducks_y_motion[duck_id] = ducks_y_motion[duck_id] / self.homography_visualization_multiplier - y_motion
            
            # ducks_x_motion = {duck_id : (-ducks_x_motion[duck_id] / self.homography_visualization_multiplier - x_motion) for duck_id in ducks_x_motion}
            # ducks_y_motion = {duck_id : (-ducks_y_motion[duck_id] / self.homography_visualization_multiplier - y_motion) for duck_id in ducks_y_motion}
            
            print(ducks_x_motion)
            
            # update ducks positions
            for duck_id in self.ducks_positions:
                self.ducks_current_x[duck_id] = self.ducks_current_x[duck_id] + ducks_x_motion[duck_id]
                self.ducks_current_y[duck_id] = self.ducks_current_y[duck_id] + ducks_y_motion[duck_id]
                duck_current_position = [self.ducks_current_x[duck_id], self.ducks_current_y[duck_id]]
                self.ducks_positions[duck_id].append(duck_current_position)
            
            for ax, axbackground in zip(self.axs, self.axs_background):
                self.fig.canvas.restore_region(axbackground)
                
            frame = self.draw_duck_bboxes(frame, duck_bboxes)
            frame = self.draw_duck_centroids(frame, duck_centroids)
            
            birds_eye_view = self.draw_duck_bboxes(birds_eye_view, warped_bboxes)
            birds_eye_view = self.draw_duck_centroids(birds_eye_view, warped_centroids)
            
            self.orig_img_show.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.birds_eye_show.set_data(cv2.cvtColor(birds_eye_view, cv2.COLOR_BGR2RGB))
            self.line_detection_show.set_data(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
            self.orb_show.set_data(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB))
            
            self.cameraman_position_plot.set_xdata([pos[0] for pos in self.cameraman_positions])
            self.cameraman_position_plot.set_ydata([pos[1] for pos in self.cameraman_positions])
            # adjust the limits
            self.axs[4].set_xlim(min([pos[0] for pos in self.cameraman_positions]) -0.2, max([pos[0] for pos in self.cameraman_positions]) + 0.2)
            self.axs[4].set_ylim(min([pos[1] for pos in self.cameraman_positions]) - 0.2, max([pos[1] for pos in self.cameraman_positions]) + 0.2)
            
            # for every duck id, add a different plot line
            for duck_id in self.ducks_positions:
                # 0-255 to 0-1
                plt_color = self.track_colors[duck_id] / 255
                self.ducks_positions_plot.plot([pos[0] for pos in self.ducks_positions[duck_id]], [pos[1] for pos in self.ducks_positions[duck_id]], color=plt_color)
                # plot duck current position
                self.ducks_positions_plot.plot(self.ducks_current_x[duck_id], self.ducks_current_y[duck_id], 'o', color=plt_color)
            
            self.axs[0].draw_artist(self.orig_img_show)
            self.axs[1].draw_artist(self.birds_eye_show)
            self.axs[2].draw_artist(self.line_detection_show)
            self.axs[3].draw_artist(self.orb_show)
            self.axs[4].draw_artist(self.cameraman_position_plot)
            self.axs[5].draw_artist(self.ducks_positions_plot)  
                
            start_time = time.time()
            show_img = self.plt_to_cv2(self.fig)
            print(f"Time took: {time.time()-start_time}")
            
            for ax in self.axs:
                self.fig.canvas.blit(ax.bbox)
                
            self.fig.canvas.flush_events()

            cv2.imshow("Analysis", show_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            prev_frame = frame
            prev_line_list = line_list
            prev_line_image = line_image
            prev_duck_bboxes = duck_bboxes
            prev_duck_centroids = duck_centroids
            prev_warped_bboxes = warped_bboxes
            prev_warped_centroids = warped_centroids
            
            if self.save_video != '':
                print(f"Show img shape: {show_img.shape}")
                # show_img = imutils.resize(show_img, width=self.FRAME_WIDTH*2)
                show_img = cv2.resize(show_img, (self.FIG_SIZE_X*100, self.FIG_SIZE_Y*100))
                self.out.write(show_img)

            frame_count += 1

            
        self.cap.release()
        if self.save_video != '':
            self.out.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track cameraman position')
    # path to save video in --save
    parser.add_argument('--save', type=str, default='', help='path to save the video')
    args = parser.parse_args()
    save_path = args.save
    
    print("Will save to path: ", save_path)
    
    CameramanPositionTracker(SOURCE, CALIBRATION_2D_POINTS_FILE, CALIBRATION_3D_POINTS_FILE, save_video=save_path)