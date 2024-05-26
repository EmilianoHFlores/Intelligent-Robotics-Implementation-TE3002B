# load the 3d image generated from the depth anything model and show it in a 3d plot
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.mini_segment_anything import build_sam_yoso_r50
from segment_anything import sam_model_registry, SamPredictor

import argparse

# save argument
parser = argparse.ArgumentParser()
parser.add_argument('--save', type=str, default='')

save_dir = parser.parse_args().save

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# sam-mini
sam_checkpoint = './sam_yoso_r50_13a999.pth'
sam_mini = build_sam_yoso_r50(checkpoint=sam_checkpoint).to('cuda')
predictor_mini = SamPredictor(sam_mini)

# Load the 3d image
depth_video = "duck_videos/depth_video.avi"
rgb_video = "duck_videos/rgb_video.avi"

depth_video = cv2.VideoCapture(depth_video)
rgb_video = cv2.VideoCapture(rgb_video)

FIGSIZE = 10

plt.ion()
fig = plt.figure(figsize=(10, 15))
ax1 = plt.subplot(1, 3, 1)
ax1.set_title('Segmented RGB Image')
plt.axis('on')
ax2 = plt.subplot(1, 3, 2)
ax2.set_title('Depth Image')
plt.axis('on')
ax3 = fig.add_subplot(122, projection='3d')
# set axis to be viewed from the top in 45 degrees
ax3.view_init(105, 45, 45)
ax3.set_title('3D Depth Image')
plt.tight_layout()

if save_dir:
    out = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'XVID'), 30, (FIGSIZE*100, FIGSIZE*100))

while True:
    ret1, frame = depth_video.read()
    ret2, rgb_frame = rgb_video.read()

    if not ret1 or not ret2:
        break
    
    # segment the image
    input_box = np.array([0, 0, rgb_frame.shape[1], rgb_frame.shape[0]])

    input_point = np.array([[rgb_frame.shape[1]//2, rgb_frame.shape[0]//2]])
    input_label = np.array([1])

    predictor_mini.set_image(frame)
    masks, scores, logits = predictor_mini.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=input_box[None, :],
        multimask_output=False,
    )

    
    ax1.imshow(rgb_frame)
    ax2.imshow(frame)
    show_mask(masks, ax1)
    show_points(input_point, input_label, ax1)
    # show_box(input_box, plt.gca())
    

    # Add your code here for the second subplot


    # make the inverse of this depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    depth = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x, y, z, colors = [], [], [], []

    DOWNSAMPLE = 0.2 # downsample the image to make it easier to plot, to a percentage of the original size

    step = 1 / DOWNSAMPLE

    for i in range(0, depth.shape[0], int(step)):
        for j in range(0, depth.shape[1], int(step)):
            # if point is not in the mask, skip it
            if not masks[0][i][j]:
                continue
            x.append(i)
            y.append(j)
            z.append(depth[i][j])
            # transformed from 0-255 to 0-1
            colors.append([rgb_frame[i][j][0] / 255, rgb_frame[i][j][1] / 255, rgb_frame[i][j][2] / 255])

    max_z = max(z)
    # # extend so that the image in z has a simmetrical replica joined to it (like if completing a cylinder)
    # for i in range(0, depth.shape[0], int(step)):
    #     for j in range(0, depth.shape[1], int(step)):
    #         if not masks[0][i][j]:
    #             continue
    #         x.append(i)
    #         y.append(j)
    #         z.append(max_z + (0.1*max_z - depth[i][j]))
    #         colors.append([rgb_frame[i][j][0] / 255, rgb_frame[i][j][1] / 255, rgb_frame[i][j][2] / 255])

    print(len(x), len(y), len(z))

    # Show the 3d image in a 3d plot
    ax3.scatter(x, y, z, c=colors, marker='o')

    # update the plot
    if save_dir:
        cv_image = np.array(fig.canvas.renderer.buffer_rgba())
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
        cv_image = cv2.resize(cv_image, (FIGSIZE*100, FIGSIZE*100))
        out.write(cv_image)
    plt.draw()
    plt.pause(0.001)
    ax3.clear()
    ax2.clear()
    ax1.clear()
    
    # Compare this snippet from vision/Depth-Anything/run-video.py:

