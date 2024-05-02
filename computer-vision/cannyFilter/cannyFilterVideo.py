# Apply a made from scratch canny filter to an image
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt 
import math
import time

SHOW_IMAGE_RESOLUTION = 500
SOURCE = "video_test.mp4"
SCRATCH_IMPLEMENTATION = True

def cannyFilter(image):
    # Convert the image to grayscale
    start_time = time.time()
    gray = grayImage(image)
    print(f"Gray image time: {time.time() - start_time}")
    # Apply a gaussian blur to the image
    start_time = time.time()
    blur = gaussianBlur(gray)
    print(f"Gaussian blur time: {time.time() - start_time}")
    # Apply the sobel operator to the image
    start_time = time.time()
    sobel, direction = sobelOperator(blur)
    print(f"Sobel operator time: {time.time() - start_time}")
    # Apply non-maximum suppression to the image
    start_time = time.time()
    nonMax = nonMaximumSuppression(sobel, direction)
    print(f"Non-maximum suppression time: {time.time() - start_time}")
    # Apply hysteresis thresholding to the image
    start_time = time.time()
    hysteresis = hysteresisThresholding(nonMax)
    print(f"Hysteresis thresholding time: {time.time() - start_time}")
    print("------------------------------------")
    
    return gray, blur, sobel, nonMax, hysteresis

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
    # grayImage = np.sum(grayImage, axis=2) / 3
    grayImage = r_coeff * grayImage[:, :, 2] + g_coeff * grayImage[:, :, 1] + b_coeff * grayImage[:, :, 0]
    # Notice the resulting numpy array only has one channel now, so shape is (height, width) instead of (height, width, 3)
    return grayImage
    

def gaussianBlur(image, kernelSize=5, sigma=1):
    
    gaussianImage = image.copy()
    # Create a kernel for the gaussian blur
    kernel = np.zeros((kernelSize, kernelSize))
    def gaussian(x, y, sigma):
        # Gaussian function to fill the kernel, uses the formula for a 2D gaussian distribution
        return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # Fill the kernel with values
    kernelSum = 0
    for i in range(kernelSize):
        for j in range(kernelSize):
            cell = gaussian(i - kernelSize // 2, j - kernelSize // 2, sigma)
            kernel[i, j] = cell
            kernelSum += cell
    # Normalize the kernel
    kernel /= kernelSum
    
    # Apply the kernel to the image with convolution, using the cv2.filter2D function
    # convolution can be applied manually like:
    # for i in range(0, image.shape[0] - kernelSize):
    #     for j in range(0, image.shape[1] - kernelSize):
    #         gaussianImage[i, j] = np.sum(image[i:i + kernelSize, j:j + kernelSize] * kernel)
    # but cv2.filter2D is faster as it uses M*N(K+L) operations instead of M*N*K*L operations
    gaussianImage = cv2.filter2D(gaussianImage, -1, kernel)
    
    return gaussianImage

def sobelOperator(image):
    # Create a kernel for the sobel operator
    sobelImage = image.copy()
    # sobel operators
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # Apply the sobel operator to the image with convolution
    sobelXImage = cv2.filter2D(sobelImage, -1, sobelX)
    sobelYImage = cv2.filter2D(sobelImage, -1, sobelY)
    # Calculate the gradient magnitude
    sobelImage = np.sqrt(sobelXImage ** 2 + sobelYImage ** 2)
    # Normalize the gradient magnitude
    sobelImage = (sobelImage / np.max(sobelImage) * 255).astype(np.uint8)
    direction = np.arctan2(sobelYImage, sobelXImage)
    
    return sobelImage, direction

def nonMaximumSuppression(image, direction):
    # Apply non-maximum suppression to the image, which thins the edges by setting pixels to zero if they are not the maximum in the direction of the gradient
    # this means that these pixels are not actually the edge but a part of it
    nonMaxImage = image.copy()
    # convert to degrees
    direction = np.rad2deg(direction)
    # make all negative angles positive by adding 180 degrees
    direction[direction < 0] += 180
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            angle = direction[i, j]
            # this checks for pixel at either the horizontal, vertical or diagonal directionsss
            # check if the angle is within the 0-45 degree range
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                if (image[i, j] < image[i, j + 1]) or (image[i, j] < image[i, j - 1]):
                    nonMaxImage[i, j] = 0
            # check if the angle is within the 45-90 degree range
            elif (22.5 <= angle < 67.5):
                if (image[i, j] < image[i - 1, j + 1]) or (image[i, j] < image[i + 1, j - 1]):
                    nonMaxImage[i, j] = 0
            # check if the angle is within the 90-135 degree range
            elif (67.5 <= angle < 112.5):
                if (image[i, j] < image[i - 1, j]) or (image[i, j] < image[i + 1, j]):
                    nonMaxImage[i, j] = 0
            # check if the angle is within the 135-180 degree range
            elif (112.5 <= angle < 157.5):
                if (image[i, j] < image[i - 1, j - 1]) or (image[i, j] < image[i + 1, j + 1]):
                    nonMaxImage[i, j] = 0
    
    return nonMaxImage

def hysteresisThresholding(image, lowThreshold=0.02, highThreshold=0.15):
    # Apply hysteresis thresholding to the image, which uses two thresholds to determine strong, weak, and non-relevant pixels
    hysteresisImage = image.copy()
    
    tresholdHigh = highThreshold * np.max(hysteresisImage)
    tresholdLow = lowThreshold * np.max(hysteresisImage)
    
    # value in between the high and low threshold, 255 is white, 50 is a dark gray
    weak = 50
    strong = 255
    
    # set all pixels above the high threshold to strong
    hysteresisImage[hysteresisImage > tresholdHigh] = strong
    # set all pixels below the low threshold to weak
    hysteresisImage[hysteresisImage < tresholdLow] = 0
    # set all pixels in between the high and low threshold to weak
    hysteresisImage[(hysteresisImage >= tresholdLow) & (hysteresisImage <= tresholdHigh)] = weak
    
    return hysteresisImage

def joinImages(images=[], rows=0, cols=0):
    # Join images into a single image, assumes all images have the same height
    # convert all images to RGB in case they are grayscale or have only one channel
    for i in range(len(images)):
        # print(len(images[i].shape))
        if len(images[i].shape) == 2:
            images[i] = cv2.merge((images[i], images[i], images[i]))
        elif images[i].shape[2] == 1:
            images[i] = cv2.merge((images[i], images[i], images[i]))
        
    # rows takes precedence over cols
    if rows == 0 and cols == 0:
        rows = len(images)
    if rows == 0:
        rows = math.ceil(len(images) / cols)
    if cols == 0:
        cols = math.ceil(len(images) / rows)
        
    #print(f"Rows: {rows}, Cols: {cols}")
    # resize all images to the same height, so that all fit in a cv window of height height
    original_height = images[0].shape[0]
    original_width = images[0].shape[1]
    
    #print(f"Original width: {original_width}, Original height: {original_height}")
    for i in range(len(images)):
        images[i] = imutils.resize(images[i], height=original_height//rows, width=original_width//cols)
    # join the images
    result_image = np.zeros((images[0].shape[0] * rows, images[0].shape[1] * cols, 3), dtype=np.uint8)
    #print(f"Result image shape: {result_image.shape}")
    image_counter = 0
    for i in range(rows):
        row_start = images[image_counter].shape[0] * i
        for j in range(cols):
            col_start =images[image_counter].shape[1] * j
            image = images[image_counter]
            # print(f"Image {image_counter} shape: {image.shape}, row_start: {row_start}, col_start: {col_start}")
            result_image[row_start:row_start + image.shape[0], col_start:col_start + image.shape[1]] = image
            image_counter += 1
            
    return result_image


if __name__ == "__main__":
    
    # reads from video or camera
    cap = cv2.VideoCapture(SOURCE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = imutils.resize(frame, height=SHOW_IMAGE_RESOLUTION)
        if SCRATCH_IMPLEMENTATION:
            gray, blur, sobel, nonMax, hysteresis = cannyFilter(frame)
            
            show_image = joinImages([frame, gray, blur, sobel, nonMax, hysteresis], rows=2)
            show_image = imutils.resize(show_image, height=SHOW_IMAGE_RESOLUTION)
            cv2.imshow("Canny Filter", show_image)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 10, 50)
            show_image = joinImages([frame, edges], rows=1)
            show_image = imutils.resize(show_image, height=SHOW_IMAGE_RESOLUTION)
            cv2.imshow("Canny Filter", show_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()