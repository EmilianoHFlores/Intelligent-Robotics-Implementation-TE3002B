# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame

import cv2
import numpy as np

def infer_process(predictions: dict, video_frame: VideoFrame):
    print(f"Frame ID: {video_frame.frame_id}")
    
    frame = video_frame.image.copy()
    for prediction in predictions["predictions"]:
        x = int(prediction["x"])
        y = int(prediction["y"])
        width = int(prediction["width"])
        height = int(prediction["height"])
        
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
        
        label = f"{prediction['class']} {prediction['confidence']:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    

# uses a model uploaded to Roboflow to detect ducks in a video
pipeline = InferencePipeline.init(
    model_id="ducks_tec_detection/3", # Roboflow model to use
    video_reference="ducks.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=infer_process,
)

pipeline.start()
pipeline.join()

cv2.destroyAllWindows()