import cv2
import numpy as np
from ultralytics import YOLO
from kalman_tracker import KalmanTracker  # Your class from kalman_tracker.py

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load video
video_path = r"D:\XAMPP\htdocs\Projects\trafficAI\data\raw\roadTrafficVideo_trimmed.mp4"
cap = cv2.VideoCapture(video_path)

# Define polygonal zones
zones = {
    "Zone A": np.array([[198, 324], [515, 326], [501, 200], [286, 197], [197, 324]], np.int32),
    "Zone B": np.array([[534, 364], [730, 358], [771, 594], [506, 595], [534, 364]], np.int32),
    "Zone C": np.array([[1026, 395], [1195, 470], [1116, 624], [850, 520], [1027, 396]], np.int32)
}

# Initialize tracker and zone counters
tracker = KalmanTracker()
zone_entry_counts = {zone: 0 for zone in zones}
zone_visits = {zone: set() for zone in zones}  # Stores track IDs per zone

# Helper functions
def draw_zones(frame):
    for name, poly in zones.items():
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_in_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1281, 720))
    detections = model(frame, verbose=False)[0]

    # Get centers of detected vehicles
    vehicle_centers = []
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:  # Vehicle classes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = box_center((x1, y1, x2, y2))
            vehicle_centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)

    # Update tracker with detected centers
    tracked_objects = tracker.update(vehicle_centers)

    # Check zone entries
    for obj_id, (x, y) in tracked_objects.items():
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {obj_id}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for zone_name, polygon in zones.items():
            if point_in_zone((x, y), polygon) and obj_id not in zone_visits[zone_name]:
                zone_entry_counts[zone_name] += 1
                zone_visits[zone_name].add(obj_id)

    # Draw zones and overlay counts
    draw_zones(frame)
    y_offset = 20
    for zone, count in zone_entry_counts.items():
        cv2.putText(frame, f"{zone}: {count} vehicles", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    cv2.imshow("Smart Traffic Counter (Kalman)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
