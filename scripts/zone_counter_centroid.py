import cv2
import numpy as np
from ultralytics import YOLO
from centroid_tracker import CentroidTracker

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = r"D:\XAMPP\htdocs\Projects\trafficAI\data\raw\roadTrafficVideo_trimmed.mp4"
cap = cv2.VideoCapture(video_path)

# Define polygon zones
#zones = {
    #"Zone A": np.array([[753, 654], [1072, 680], [1101, 406], [979, 392], [754, 654]], np.int32),
    "#Zone B": np.array([[712, 474], [507, 458], [124, 587], [341, 668], [707, 474]], np.int32),
#}

#def draw_zones(frame):
    for name, poly in zones.items():
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#def get_zone(point):
    #for name, poly in zones.items():
   #     if cv2.pointPolygonTest(poly, point, False) >= 0:
  #          return name
 #   return None

# Initialize tracker and zone memory
#ct = CentroidTracker(max_disappeared=50, iou_threshold=0.2)
#id_zone_memory = {}
#zone_entry_counts = {name: 0 for name in zones}

#while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1281, 720))
    results = model(frame, verbose=False)[0]

    boxes = []
    classes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            classes.append(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    tracked_objects = ct.update(boxes, classes)

    for object_id, (x, y) in tracked_objects.items():
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(frame, f"ID {object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        current_zone = get_zone((x, y))
        last_zone = id_zone_memory.get(object_id)

        if current_zone and current_zone != last_zone:
            zone_entry_counts[current_zone] += 1
            id_zone_memory[object_id] = current_zone

        # Draw trail
        if object_id in ct.tracks:
            trail = ct.tracks[object_id]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 2)

    draw_zones(frame)

    # Display zone counts
    y_offset = 20
    for zone, count in zone_entry_counts.items():
        cv2.putText(frame, f"{zone}: {count} vehicles", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    cv2.imshow("Traffic Detection with Zones", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
