import cv2
import numpy as np
import os

import cv2.aruco as aruco
import threading
import socket
import time

IP = "127.0.0.1"
PORT = 5204

PUBLISH_RATE = 30

ENABLE_SOCKET = True

class arucoDetector:
    def __init__(self, markerSize=6, totalMarkers=250):
        self.markerSize = markerSize
        self.totalMarkers = totalMarkers
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def findMarkers(self, img, draw=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxs, ids, rejected = self.detector.detectMarkers(gray)
        return bboxs, ids, rejected

    def getPerpendicularAngle(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        angle = np.arctan2(y2 - y1, x2 - x1)
        return angle

    def processImage(self, img):
        bboxes, ids, rejected = self.findMarkers(img)
        angle = None
        centroid = [-1, -1]
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
                    angle = self.getPerpendicularAngle(bbox[0], bbox[1])
                    print(angle)
                    # draw arrow from center of bbox to the center of the screen
                    center = (int(sum([x[0] for x in bbox]) / 4), int(sum([x[1] for x in bbox]) / 4))
                    centroid = center
                    cv2.circle(img, center, 5, (0, 0, 255), cv2.FILLED)
                    arrow_tip = (int(center[0] + 50 * np.cos(angle)), int(center[1] + 50 * np.sin(angle)))
                    cv2.arrowedLine(img, center, arrow_tip, (0, 0, 255), 2)
                    
                    # invert centroid x (as the camera is inverted)
                    centroid = [img.shape[1] - centroid[0], centroid[1]]

        return img, centroid, angle

class arucoPublisher:
    def __init__(self):
        self.IP = IP
        self.PORT = PORT
        self.RATE = PUBLISH_RATE
        
        self.centroid = [-1, -1]
        self.angle = None
        
        if ENABLE_SOCKET:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("Waiting for connection...")
            self.server.bind((self.IP, self.PORT))
            self.server.listen(1)
            self.client, addr = self.server.accept()
            print(f"Connection from {addr}")
            publish_thread = threading.Thread(target=self.publish)
            self.active = True
            publish_thread.start()
        
        self.main()
        
    def publish(self):
        last_publish_time = time.time()
        while self.active:
            current_time = time.time()
            if current_time - last_publish_time >= 1 / self.RATE:
                if self.angle is not None:
                    data = f"{self.centroid[0]},{self.centroid[1]},{self.angle}"
                    self.client.send(data.encode())
                    print(f"Published: {data}")
                    last_publish_time = current_time
    
    def main(self):
        cap = cv2.VideoCapture(0)
        detector = arucoDetector()
        
        while self.active:
            success, img = cap.read()
            img, self.centroid, self.angle = detector.processImage(img)
            
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                self.active = False
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    arucoPublisher()

if __name__ == "__main__":
    main()