import numpy as np  
import time
import cv2
import imutils
cv2.namedWindow('matches', 0)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
cv2.namedWindow('motion', cv2.WINDOW_NORMAL)

FRAME_HEIGHT = 7207


video = cv2.VideoCapture("ducks.mp4")

ret, prev_frame = video.read()
width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.resizeWindow('matches', prev_frame.shape[1], prev_frame.shape[0])
cv2.resizeWindow('motion', prev_frame.shape[1], prev_frame.shape[0])


# Camera intrinsic parameters
fx = 1762.3818658905311
fy = 1771.882532753107
x0 = width / 2
y0 = height / 2

K = np.array([[fx, 0, x0],
                [0, fy, y0],
                [0, 0, 1]])

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
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def draw_matches(self, img1, kp1, img2, kp2, matches):
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img3
    
    def calculate_motion(self, kp1, kp2, matches):
        # get motion from one frame to another  
        print("--------------------")   
        matched_points_1 = np.array([kp1[match.queryIdx].pt for match in matches])
        matched_points_2 = np.array([kp2[match.trainIdx].pt for match in matches])

        if len(matched_points_1) < 8 or len(matched_points_2) < 8:
            return None
        
        # print(f"matched_points_1: {matched_points_1}")
        # print(f"matched_points_2: {matched_points_2}")
        # draw each in a different plot, match the points with the same index with color
        fig = plt.figure(figsize=(15,15))
        plt.tight_layout()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        # Assigning colors to each point
        colors = np.random.rand(len(matched_points_1), 3)
        # Plotting the first set of points
        ax1.scatter(matched_points_1[:, 0], matched_points_1[:, 1], c=colors)
        ax1.set_title('First set of points')
        # Plotting the second set of points
        ax2.scatter(matched_points_2[:, 0], matched_points_2[:, 1], c=colors)
        ax2.set_title('Second set of points')
        
        center_1 = np.mean(matched_points_1, axis=0)
        center_2 = np.mean(matched_points_2, axis=0)
        
        # Plotting the centers of the points
        ax1.scatter(center_1[0], center_1[1], c='r', marker='x')
        ax2.scatter(center_2[0], center_2[1], c='r', marker='x')
        
        # center each set of points at the origin
        centered_points_1 = matched_points_1 - center_1
        centered_points_2 = matched_points_2 - center_2
        
        # plot the centered points
        ax3 = fig.add_subplot(223)
        ax3.scatter(centered_points_1[:, 0], centered_points_1[:, 1], c=colors)
        ax3.scatter(centered_points_2[:, 0], centered_points_2[:, 1], c=colors)
        ax3.scatter(0,0, c='r', marker='x')
        
        # Estimate the essential matrix from point to point correspondences
        E, mask = cv2.findEssentialMat(centered_points_1, centered_points_2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        inliers1 = centered_points_1[mask]
        inliers2 = centered_points_2[mask]

        # Decompose the essential matrix to get the relative rotation and translation
        _, R, t, mask = cv2.recoverPose(E, centered_points_1, centered_points_2, K)
        # _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, K)

        # R, _ = cv2.Rodrigues(R)

        recovered_rotation_euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        
        print(f"recovered_rotation_euler: {recovered_rotation_euler}")
        print(f"t: {t}")
        print(f"R: {R}")

        # calculate the rotation matrix
        H = np.dot(centered_points_1.T, centered_points_2)
        
        # calculate the singular value decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # calculate the rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # calculate the translation vector
        t = center_2 - np.dot(R, center_1)
        print("--------------------")
        print(f"t: {t}")
        print(f"R: {R}")

        # calculate the transformed points
        transformed_points = np.dot(centered_points_1, R.T) + t
        
        # plot the transformed points
        ax4 = fig.add_subplot(224)
        ax4.scatter(transformed_points[:, 0], transformed_points[:, 1], c=colors)
        ax4.scatter(centered_points_2[:, 0], centered_points_2[:, 1], c=colors)
        
        # plot the center of the points
        ax4.scatter(0,0, c='r', marker='x')
        
        # angle of rotation
        angle = np.arctan2(R[1, 0], R[0, 0])
        # x translation
        t[0]
        # y translation
        t[1]
        
        # print(f"angle: {angle}")
        # print(f"t: {t}")
        print("--------------------")
        
        return fig
        
orb_points = OrbPoints()
while ret:
    ret, frame = video.read()
    if not ret:
        break

    kp1, des1 = orb_points.detect_and_compute(prev_frame)
    kp2, des2 = orb_points.detect_and_compute(frame)

    matches = orb_points.match_descriptors(des1, des2)
    fig = orb_points.calculate_motion(kp1, kp2, matches)

    fig.canvas.draw()
    print("transformnig to array")
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGRA2RGB)
    print(img_plot.shape)

    img3 = orb_points.draw_matches(prev_frame, kp1, frame, kp2, matches)
    print("drawn matches")
    
    #cv2.imshow("motion", img_plot)
    cv2.imshow("matches", img3)
    cv2.imshow("motion", img_plot)
    # wait until a key is pressed to continue
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
    prev_frame = frame

cv2.destroyAllWindows()
video.release()