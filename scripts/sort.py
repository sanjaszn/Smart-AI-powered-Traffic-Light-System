import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.track_count = 0

    def update(self, detections):
        """
        Updates the trackers with new detections and returns the updated tracks.
        Detections are expected to be in the form [x1, y1, x2, y2, confidence]
        """
        if len(detections) == 0:
            return np.empty((0, 5))

        updated_tracks = []

        # Convert detections to the format [x1, y1, x2, y2]
        dets = np.array(detections)[:, :4]

        # Update each tracker with the new detections
        for tracker in self.trackers:
            tracker.predict()

        # Perform the Hungarian algorithm to find the best matches
        cost_matrix = self._compute_cost_matrix(dets)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.iou_threshold:
                self.trackers[j].update(dets[i])
                updated_tracks.append(self.trackers[j].get_state())

        # Remove trackers that have not been updated in a while
        self.trackers = [tracker for tracker in self.trackers if tracker.age < self.max_age]

        return np.array(updated_tracks)

    def _compute_cost_matrix(self, detections):
        """
        Computes the cost matrix for tracking, typically using IOU or Euclidean distance.
        """
        cost_matrix = np.zeros((len(detections), len(self.trackers)))

        for i, detection in enumerate(detections):
            for j, tracker in enumerate(self.trackers):
                cost_matrix[i, j] = tracker.compute_cost(detection)

        return cost_matrix
