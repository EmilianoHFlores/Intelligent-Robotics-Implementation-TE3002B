import cv2
import numpy as np


def blue_segmentation(frame, hsv_treshold=[(0, 0, 0), (140, 255, 255)]):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_treshold[0], hsv_treshold[1])
    blue = cv2.bitwise_and(frame, frame, mask=mask)
    return blue

def binary_conversion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    # make binary image to 3 channels
    binary_image = cv2.merge((binary_image, binary_image, binary_image))
    return binary_image

def get_circles(frame):
    frame = cv2.GaussianBlur(frame, (9, 9), 2)
    # binary to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1.3, 50, param1=250, param2=100, minRadius=5, maxRadius=0)
    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    except:
        pass
    return frame, circles

cap = cv2.VideoCapture(0)

while True:
    # hsv treshold for blue color
    hsv_treshold = [(100, 100, 100), (140, 255, 255)]
    ret, frame = cap.read()
    treshold_image = blue_segmentation(frame, hsv_treshold)
    binary_image = binary_conversion(treshold_image)
    binary_circles, circles = get_circles(binary_image)
    result_image = frame.copy()
    try:
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(result_image,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(result_image,(i[0],i[1]),2,(0,0,255),3)
    except:
        pass
    cv2.imshow('frame', frame)
    cv2.imshow('treshold_image', treshold_image)
    cv2.imshow('binary_image', binary_image)
    cv2.imshow('binary_circles', binary_circles)
    cv2.imshow('result_image', result_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.closeAllWindows()
cap.release()