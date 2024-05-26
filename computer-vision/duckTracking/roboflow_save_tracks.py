# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame
from motpy import Detection, MultiObjectTracker
import cv2
import numpy as np
import pickle

MAX_TRACKS = 100

# generate MAX_TRACKS unique colors for each track
colors = np.random.randint(0, 255, size=(MAX_TRACKS, 3), dtype="uint8")

first_track = True

track_dict = {}
tracks_to_follow = []

# list to save all bboxes and ids from each frame
frame_tracks = []
frame_count = 0

def infer_process(predictions: dict, video_frame: VideoFrame):
    global first_track, track_dict, tracks_to_follow, frame_tracks, frame_count
    print(f"Frame ID: {video_frame.frame_id}")
    
    frame = video_frame.image.copy()
    tracker_detections = []
    for prediction in predictions["predictions"]:
        x = int(prediction["x"])
        y = int(prediction["y"])
        width = int(prediction["width"])
        height = int(prediction["height"])
        
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
        
        label = f"{prediction['class']} {prediction['confidence']:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        detection_box = np.array([x1, y1, x2, y2])
        
        tracker_detections.append(Detection(box=detection_box))
        
    _ = multi_tracker.step(tracker_detections)
    tracks = multi_tracker.active_tracks()
    if first_track:
        tracks_to_follow = [track.id for track in tracks]
        for track in tracks:
            track_dict[track.id] = []
        first_track = False
    frame_track = {}
    for i, track in enumerate(tracks):
        # cv2.putText(frame, f"ID: {track.id}", (int(track.box[0]), int(track.box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)
        cv2.putText(frame, f"ID: {track.id}", (int(track.box[0]), int(track.box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)
        cv2.rectangle(frame, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), colors[i].tolist(), 2)
        
        
        if track.id in tracks_to_follow:
            # print("reencountered track: ", track.id)
            track_dict[track.id].append([int(track.box[0]), int(track.box[1]), int(track.box[2]), int(track.box[3])])
            frame_track[track.id] = [int(track.box[0]), int(track.box[1]), int(track.box[2]), int(track.box[3])]
    
    print("Appending frame track: ", frame_track)
    frame_tracks.append(frame_track)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    

SOURCE = "stabilized_video_ducks.avi"
video = cv2.VideoCapture(SOURCE)
fps = video.get(cv2.CAP_PROP_FPS)
multi_tracker = MultiObjectTracker(dt=1/fps, tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': 0.25,
                            'multi_match_min_iou': 0.93})

# uses a model uploaded to Roboflow to detect ducks in a video
pipeline = InferencePipeline.init(
    model_id="ducks_tec_detection/3", # Roboflow model to use
    video_reference=SOURCE, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=infer_process,
)

pipeline.start()
pipeline.join()
    
pickle.dump(frame_tracks, open("frame_tracks.pkl", "wb"))
cv2.destroyAllWindows()