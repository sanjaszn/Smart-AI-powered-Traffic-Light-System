#MAIN PROJECT SCRIPT
import cv2
import numpy as np
import sys
import os
from argparse import Namespace
from ultralytics import YOLO

# Extend sys path to access ByteTrack
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bytetrack'))
from yolox.tracker.byte_tracker import BYTETracker

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
video_path = r"D:\XAMPP\htdocs\Projects\trafficAI\data\raw\roadTrafficVideo_trimmed.mp4"
cap = cv2.VideoCapture(video_path)

# Get original video resolution
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame.")
original_height, original_width = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Resize settings for display only
display_size = (1280, 720)

# Define zones in original resolution
zones = {
    "Zone A": np.array([[534, 906], [693, 678], [1563, 690], [1557, 924], [537, 900]], np.int32),
    "Zone B": np.array([[1542, 1041], [1527, 1233], [2235, 1209], [2187, 1020], [1542, 1038]], np.int32),
    "Zone C": np.array([[2613, 1314], [2850, 1152], [3672, 1611], [3564, 1833], [2613, 1314]], np.int32)
}

# Init BYTETracker
args = Namespace(
    track_thresh=0.5, match_thresh=0.8, track_buffer=30,
    aspect_ratio_thresh=1.6, min_box_area=10, mot20=False
)
tracker = BYTETracker(args=args)

zone_entry_counts = {zone: 0 for zone in zones}
zone_visits = {zone: set() for zone in zones}

def draw_zones(frame):
    for name, poly in zones.items():
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def box_intersects_zone(box, polygon):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
        return True
    return False

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    full_frame = frame.copy()

    # Run detection
    results = model(full_frame, verbose=False)[0]

    # Extract vehicle detections
    vehicle_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            vehicle_boxes.append([x1, y1, x2, y2, conf])

    if not vehicle_boxes:
        print(f"Frame {frame_id}: No vehicle detections.")
    else:
        print(f"Frame {frame_id}: {len(vehicle_boxes)} detections.")

    vehicle_boxes = np.array(vehicle_boxes, dtype=np.float32) if vehicle_boxes else np.empty((0, 5), dtype=np.float32)

    # Update tracker
    tracked_objects = tracker.update(
    vehicle_boxes,
    [original_height, original_width],
    (original_height, original_width)
    )

    print(f"Frame {frame_id}: {len(tracked_objects)} tracks.")

    for track in tracked_objects:
        x1, y1, w, h = map(int, track.tlwh)
        x2, y2 = x1 + w, y1 + h
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        track_id = track.track_id

        # Draw bounding box and ID
        cv2.rectangle(full_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(full_frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(full_frame, (cx, cy), 5, (0, 0, 255), -1)

        # Check for zone entry
        for zone_name, polygon in zones.items():
            if box_intersects_zone((x1, y1, x2, y2), polygon) and track_id not in zone_visits[zone_name]:
                zone_entry_counts[zone_name] += 1
                zone_visits[zone_name].add(track_id)
                cv2.putText(full_frame, f"In {zone_name}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Optionally visualize raw detection boxes (red)
    for box in vehicle_boxes:
        x1, y1, w, h, conf = map(int, box)
        cv2.rectangle(full_frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 1)

    draw_zones(full_frame)

    y_offset = 20
    for zone, count in zone_entry_counts.items():
        cv2.putText(full_frame, f"{zone}: {count} vehicles", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    display_frame = cv2.resize(full_frame, display_size)
    cv2.imshow("Smart Traffic Counter", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
