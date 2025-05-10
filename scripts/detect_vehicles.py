import cv2
import os

# Path setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frames_path = os.path.join(base_dir, "data", "processed", "frames")
output_path = os.path.join(base_dir, "data", "processed", "detected")

# Load Haar cascade classifier
car_cascade_path = os.path.join(base_dir, "data", "models", "cars.xml")
car_cascade = cv2.CascadeClassifier(car_cascade_path)

# Make sure output folder exists
os.makedirs(output_path, exist_ok=True)

# Loop through frames
for filename in os.listdir(frames_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(frames_path, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect cars
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)

        # Draw rectangles
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save result
        output_file = os.path.join(output_path, filename)
        cv2.imwrite(output_file, img)

print("✅ Detection complete. Check the 'detected' folder.")
