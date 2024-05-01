# Segmentates the largest white area in an image, including its inside using convex hull

import cv2
import numpy as np
import imutils
from scipy.spatial import ConvexHull
import time
import matplotlib.pyplot as plt
import skimage
from skimage import measure


def whiteSegmentation(image, hsv_lower, hsv_upper):
    """ Segmentates the largest white area in an image, including its inside using convex hull
    Args:
        image: The input image numpy array
        hsv_lower: The lower HSV values for the white color
        hsv_upper: The upper HSV values for the white color
    Returns:
        whiteSegmentation: The image with the white area segmented
    """
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask for the white color
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    # Apply the mask to the image
    whiteSegmentation = cv2.bitwise_and(image, image, mask=mask)
    
    return whiteSegmentation

def filter_biggest_blob(image):
    """"""
    # convert to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(binary_image.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(binary_image, out)
    # labels_mask = measure.label(binary_image)                       
    # regions = measure.regionprops(labels_mask)
    # regions.sort(key=lambda x: x.area, reverse=True)
    # if len(regions) > 1:
    #     for rg in regions[1:]:
    #         labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    # labels_mask[labels_mask!=0] = 1
    # mask = labels_mask
    # print(mask)
    # result = cv2.bitwise_and(binary_image, mask)
    return out


def filter_biggest_blob(image):
    """"""
    # convert to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image+
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(binary_image.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(binary_image, out)
    # labels_mask = measure.label(binary_image)                       
    # regions = measure.regionprops(labels_mask)
    # regions.sort(key=lambda x: x.area, reverse=True)
    # if len(regions) > 1:
    #     for rg in regions[1:]:
    #         labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    # labels_mask[labels_mask!=0] = 1
    # mask = labels_mask
    # print(mask)
    # result = cv2.bitwise_and(binary_image, mask)
    return out

if __name__ == "__main__":
    # Load the image
    cap = cv2.VideoCapture(0)
    # Define the lower and upper HSV values for white
    hsv_lower = np.array([0, 0, 200])
    hsv_upper = np.array([180, 25, 255])
    # Segmentate the white area in the image
    ret, image = cap.read()
    while ret:
        print("Starting segmentation")
        start_time = time.time()
        segmentation_image = whiteSegmentation(image, hsv_lower, hsv_upper)
        contour_image = filter_biggest_blob(segmentation_image)
        
        print(f"Segmentation took {time.time() - start_time} seconds")
        # Show both images in same window
        cv2.imshow("Original Image", image)
        cv2.imshow("White Segmentation", segmentation_image)
        cv2.imshow("Biggest Contour", contour_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        ret, image = cap.read()
    cv2.destroyAllWindows()
    cap.release()
