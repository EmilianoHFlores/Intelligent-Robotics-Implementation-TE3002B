import numpy as np
from scipy.spatial import distance as dist

class DuckTracker:
    def __init__(self, max_disappeared=25, max_ducks=7):
        # Initialize the list of available IDs and the dictionaries for storing objects and disappeared counts
        self.available_ids = list(range(max_ducks))
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, bbox):
        # Assign an available ID to a new object
        object_id = self.available_ids.pop(0)
        self.objects[object_id] = bbox
        self.disappeared[object_id] = 0

    def deregister(self, object_id):
        # Remove an object and make its ID available again
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.available_ids.append(object_id)

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_bboxes = np.zeros((len(rects), 4), dtype="int")
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            input_bboxes[i] = (start_x, start_y, end_x, end_y)
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)

        if len(self.objects) == 0:
            for i in range(len(input_bboxes)):
                self.register(input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_bboxes = list(self.objects.values())
            object_centroids = np.array([
                (int((bbox[0] + bbox[2]) / 2.0), int((bbox[1] + bbox[3]) / 2.0)) for bbox in object_bboxes
            ])

            D = dist.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    if len(self.available_ids) > 0:
                        self.register(input_bboxes[col])
                    else:
                        disappeared_ids = [id for id, count in self.disappeared.items() if count > 0]
                        if disappeared_ids:
                            reassigned_id = disappeared_ids.pop(0)
                            self.objects[reassigned_id] = input_bboxes[col]
                            self.disappeared[reassigned_id] = 0

        return self.objects
