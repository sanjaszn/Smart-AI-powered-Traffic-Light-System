import cv2
import numpy as np

class KalmanTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.tracks = {}

    def _create_kalman(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        return kf

    def update(self, detections):
        updated_objects = {}
        for det in detections:
            matched = False
            for obj_id, kf in self.trackers.items():
                prediction = kf.predict()
                px, py = prediction[0], prediction[1]
                dist = np.linalg.norm(np.array([px, py]) - np.array(det))
                if dist < 50:
                    measurement = np.array([[np.float32(det[0])], [np.float32(det[1])]])
                    kf.correct(measurement)
                    updated_objects[obj_id] = (int(px), int(py))
                    self.tracks[obj_id].append((int(px), int(py)))
                    matched = True
                    break
            if not matched:
                kf = self._create_kalman(*det)
                self.trackers[self.next_id] = kf
                self.tracks[self.next_id] = [det]
                updated_objects[self.next_id] = det
                self.next_id += 1
        return updated_objects
