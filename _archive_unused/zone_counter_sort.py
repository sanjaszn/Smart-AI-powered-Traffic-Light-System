import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Import the SORT tracker

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
video_path = r"D:\XAMPP\htdocs\Projects\trafficAI\data\raw\roadTrafficVideo_trimmed.mp4"
cap = cv2.VideoCapture(video_path)

# Define polygonal zones (defined for original frame size)
zones = {
    "Zone A": np.array([[1017, 340], [1055, 102], [1271, 200], [1271, 300], [1020, 339]], np.int32),
   # "Zone B": np.array([[534, 364], [730, 358], [771, 594], [506, 595], [534, 364]], np.int32),
    #"Zone C": np.array([[1026, 395], [1195, 470], [1116, 624], [850, 520], [1027, 396]], np.int32)
}

# Initialize tracker
tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

# Zone visit tracking
zone_entry_counts = {zone: 0 for zone in zones}
zone_visits = {zone: set() for zone in zones}

# Helper functions
def draw_zones(frame):
    for name, poly in zones.items():
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def box_intersects_zone(box, polygon):
    x1, y1, x2, y2 = box
    box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    for point in box_poly:
        if cv2.pointPolygonTest(polygon, tuple(point), False) >= 0:
            return True
    return False

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the original (unresized) frame
    detections = model(frame, verbose=False)[0]

    # Collect vehicle boxes in SORT format: [x1, y1, x2, y2, conf]
    vehicle_boxes = []
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_boxes.append([x1, y1, x2, y2, 1.0])  # Dummy confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)

    # Convert to NumPy for SORT
    vehicle_boxes = np.array(vehicle_boxes)
    tracked_objects = tracker.update(vehicle_boxes)

    for track in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, track)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw ID and center
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Check zone intersection
        for zone_name, polygon in zones.items():
            if box_intersects_zone((x1, y1, x2, y2), polygon) and obj_id not in zone_visits[zone_name]:
                zone_entry_counts[zone_name] += 1
                zone_visits[zone_name].add(obj_id)
                cv2.putText(frame, f"In {zone_name}", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw zones and counts
    draw_zones(frame)
    y_offset = 20
    for zone, count in zone_entry_counts.items():
        cv2.putText(frame, f"{zone}: {count} vehicles", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    # Display
    cv2.imshow("Smart Traffic Counter (SORT)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
