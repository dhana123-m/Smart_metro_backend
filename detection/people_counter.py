from ultralytics import YOLO
import cv2
import threading
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Global variables
latest_counts = {}   # { "compartment_1": count, ... }
running_threads = {} # Track threads
running = True       # Global flag to stop all

def _video_loop(source, comp_name):
    global latest_counts, running
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source for {comp_name}")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        count = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person
                    count += 1

        latest_counts[comp_name] = count
        time.sleep(0.1)

    cap.release()

def start_video_counts(sources: dict):
    """
    Start background threads for each compartment.
    sources = { "compartment_1": "data/videos/comp1.mp4", ... }
    """
    global running_threads, running
    running = True
    for comp, src in sources.items():
        if comp not in running_threads:
            thread = threading.Thread(target=_video_loop, args=(src, comp), daemon=True)
            running_threads[comp] = thread
            thread.start()

def stop_video_counts():
    """Stop all video threads."""
    global running, running_threads
    running = False
    running_threads = {}

def get_latest_counts():
    """Return latest people counts per compartment."""
    return latest_counts

def get_occupancy(counts, coach_capacity=120):
    """Return occupancy % + status for each compartment."""
    result = {}
    for comp, count in counts.items():
        occupancy = round((count / coach_capacity) * 100, 2)
        if occupancy < 50:
            status = "Low"
        elif occupancy < 80:
            status = "Moderate"
        else:
            status = "High"
        result[comp] = {
            "people_count": count,
            "capacity": coach_capacity,
            "occupancy": occupancy,
            "status": status
        }
    return result
