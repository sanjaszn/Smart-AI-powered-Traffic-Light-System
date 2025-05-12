# 🚦 Smart AI-Powered Traffic Light System (Nairobi Prototype)

This project uses computer vision and AI to analyze real-world traffic patterns from video and simulate intelligent, adaptive traffic light behavior — starting with Nairobi's chaotic intersections.


## Current Features

- Load, process, and display traffic videos frame-by-frame
- Polygon-based multi-zone detection (user-defined zones)
- Object detection using YOLOv8
- Multi-object tracking using ByteTrack
- Vehicle counting per zone



## Next Goals

- Replace ID-based counting logic with center-point or bounding-box based strategies to reduce overcounting
- Add vehicle type classification (car, truck, motorcycle, etc.)
- Detect congestion levels per zone
- Train reinforcement learning agent to control traffic light phases
- Simulate smart light decision-making in real time
- Build admin dashboard for real-world monitoring and testing



## Tools & Stack

- Python 3.12
- OpenCV
- NumPy / Matplotlib
- Ultralytics YOLOv8
- ByteTrack
- Git, GitHub, VS Code



## Directory Structure (Core Components Only)



trafficAI/
├── data/
│   ├── raw/                 # Raw input videos
│   └── processed/           # Output: detections, frames, etc.
│       ├── detected/
│       ├── frames/
│       └── yolo\_output/
├── scripts/
│   ├── detect\_vehicles.py         # YOLOv8-based detection logic
│   ├── frame\_extractor.py         # Frame grabbing logic
│   ├── multi\_zone\_drawer.py       # Tool for user-defined zone drawing
│   ├── play\_video.py              # Simple video playback for testing
│   ├── yolo\_detect.py             # YOLO wrapper / utility
│   ├── zone\_counter\_bytetrack.py  # Main logic : zone counting + tracking
│   └── stabilize\_video.py         # video stabilizer
├── smart\_traffic\_ai/            # (Assumed core logic entrypoint) 
│   └── main.py                    # Entrypoint logic (combine modules)
├── yolov8n.pt                     # YOLOv8 model weights
├── requirements.txt
├── README.md


>> *Unused or experimental trackers like SORT, Kalman, and centroid-based logic were removed for clarity. They can be found in the `_archive_unused/` folder (if needed for future experiments).*


👤 Author

>>>**Sanja Timothy**
  
-Computer Science Student – Nairobi  
-Focused on scalable AI systems for real-world problems.



>> Notes

> Large video and virtual environment files are excluded from this repo using `.gitignore`.  
> You may need to download sample videos and set up a virtual environment before running the full pipeline.
