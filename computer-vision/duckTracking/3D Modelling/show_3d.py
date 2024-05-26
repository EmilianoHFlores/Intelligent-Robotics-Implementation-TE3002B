# load the 3d image generated from the depth anything model and show it in a 3d plot
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the 3d image
depth_video = "duck_videos/depth_video.avi"
rgb_video = "duck_videos/rgb_video.avi"

depth_video = cv2.VideoCapture(depth_video)
rgb_video = cv2.VideoCapture(rgb_video)

ret, frame = depth_video.read()
ret, rgb_frame = rgb_video.read()


# make the inverse of this depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

x, y, z, colors = [], [], [], []

DOWNSAMPLE = 0.1 # downsample the image to make it easier to plot, to a percentage of the original size

step = 1 / DOWNSAMPLE

for i in range(0, depth.shape[0], int(step)):
    for j in range(0, depth.shape[1], int(step)):
        x.append(i)
        y.append(j)
        z.append(depth[i][j])
        # transformed from 0-255 to 0-1
        colors.append([rgb_frame[i][j][0] / 255, rgb_frame[i][j][1] / 255, rgb_frame[i][j][2] / 255])

print(len(x), len(y), len(z))

# Show the 3d image in a 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c=colors, marker='o')

plt.show()
# Compare this snippet from vision/Depth-Anything/run-video.py:

