# illustrates the difference between using coefficients and using mean to convert an image to grayscale.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayImage(image):
    """ Convert the image to grayscale using the NTSC standard
    Args:
        image: The input image numpy array
    Returns:
        grayImage: The grayscale image as a numpy array
    """
    grayImage = image.copy()
    # coefficients for converting to grayscale, based on NTSC standard for converting RGB to luminance (grayscale)
    r_coeff = 0.2989
    g_coeff = 0.5870
    b_coeff = 0.1140
    # cv2 format is BGR, each channel is multiplied by its coefficient
    # without coefficient would be:
    mean_image = np.sum(grayImage, axis=2) / 3
    coeff_image = r_coeff * grayImage[:, :, 2] + g_coeff * grayImage[:, :, 1] + b_coeff * grayImage[:, :, 0]
    # Notice the resulting numpy array only has one channel now, so shape is (height, width) instead of (height, width, 3)
    return mean_image, coeff_image

# Load the image
image = cv2.imread('test.jpg')
# Convert the image to grayscale
mean_image, coeff_image = grayImage(image)
# Display the image
plt.figure(figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(mean_image, cmap='gray')
plt.title('Mean Image')
plt.subplot(1, 3, 3)
plt.imshow(coeff_image, cmap='gray')
plt.title('Coeff Image')
plt.show()
