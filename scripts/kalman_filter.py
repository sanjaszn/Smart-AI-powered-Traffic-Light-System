import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        self.kf.R = np.array([[10, 0],
                               [0, 10]])  # Measurement noise
        self.kf.P *= 1000.  # Uncertainty
        self.kf.Q = np.array([[0.1, 0, 0, 0],
                               [0, 0.1, 0, 0],
                               [0, 0.1, 0, 0],
                               [0, 0, 0, 0.1]])  # Process noise

        self.state = np.zeros(4)
        self.id = None
        self.age = 0

    def predict(self):
        self.kf.predict()

    def update(self, detection):
        self.kf.update(detection)
        self.age = 0  # Reset age after update

    def get_state(self):
        return self.state

    def compute_cost(self, detection):
        """
        Compute the cost (e.g., distance) between this tracker and a new detection.
        """
        x, y = self.state[:2]
        detection_x, detection_y = detection[:2]
        return np.linalg.norm([x - detection_x, y - detection_y])
