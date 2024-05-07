import cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(detector, img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    bboxs, ids, rejected = detector.detectMarkers(gray)
    return bboxs, ids, rejected

def getPerpendicularAngle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle = np.arctan2(y2 - y1, x2 - x1)
    return angle

def main():
    cap = cv2.VideoCapture(0)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    while True:
        success, img = cap.read()
        bboxes, ids, rejected = findArucoMarkers(detector, img)
        if len(bboxes) > 0:
            for bbox, id in zip(bboxes, ids):
                bbox = bbox[0]
                if len(bbox) > 0:
                    for i, xy in enumerate(bbox):
                        # each xy is a corner, draw lines between them
                        cv2.circle(img, (int(xy[0]), int(xy[1])), 5, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, str(i), (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    # betweeen 0 and 2
                    point0 = (int(bbox[0][0]), int(bbox[0][1]))
                    point1 = (int(bbox[1][0]), int(bbox[1][1]))
                    # cv2.line(img, point0, point1, (255, 0, 255), 2)
                    angle = getPerpendicularAngle(bbox[0], bbox[1])
                    print(angle)
                    # draw arrow from center of bbox to the center of the screen
                    center = (int(sum([x[0] for x in bbox]) / 4), int(sum([x[1] for x in bbox]) / 4))
                    cv2.circle(img, center, 5, (0, 0, 255), cv2.FILLED)
                    arrow_tip = (int(center[0] + 50 * np.cos(angle)), int(center[1] + 50 * np.sin(angle)))
                    cv2.arrowedLine(img, center, arrow_tip, (0, 0, 255), 2)
                    
                    

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()