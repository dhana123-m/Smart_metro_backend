from ultralytics import YOLO
import cv2
import threading
import time

# ----------------------------
# Load YOLO model safely
# ----------------------------
# Use weights_only=True to avoid pickling errors
model = YOLO("retrained_model.pt")

# ----------------------------
# Global variables
# ----------------------------
latest_counts = {}   # { "compartment_1": count, ... }
running_threads = {} # Track active threads
running = True       # Flag to stop threads

# ----------------------------
# Video processing loop
# ----------------------------
def _video_loop(source, comp_name):
    global latest_counts, running
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source for {comp_name}")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame)
        count = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person class
                    count += 1

        latest_counts[comp_name] = count
        time.sleep(0.1)  # small delay to reduce CPU usage

    cap.release()

# ----------------------------
# Start threads for compartments
# ----------------------------
def start_video_counts(sources: dict):
    """
    sources = {
        "compartment_1": "data/videos/comp1.mp4",
        "compartment_2": "data/videos/comp2.mp4",
        ...
    }
    """
    global running_threads, running
    running = True
    for comp, src in sources.items():
        if comp not in running_threads:
            thread = threading.Thread(target=_video_loop, args=(src, comp), daemon=True)
            running_threads[comp] = thread
            thread.start()
            print(f"[INFO] Started video thread for {comp}")

# ----------------------------
# Stop all video threads
# ----------------------------
def stop_video_counts():
    global running, running_threads
    running = False
    running_threads = {}
    print("[INFO] All video threads stopped.")

# ----------------------------
# Get latest counts
# ----------------------------
def get_latest_counts():
    """Return latest people counts per compartment."""
    return latest_counts

# ----------------------------
# Get occupancy percentage + status
# ----------------------------
def get_occupancy(counts, coach_capacity=120):
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

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    sources = {
        "compartment_1": "data/videos/comp1.mp4",
        "compartment_2": "data/videos/comp2.mp4"
    }

    start_video_counts(sources)

    try:
        while True:
            counts = get_latest_counts()
            occupancy = get_occupancy(counts)
            print(occupancy)
            time.sleep(2)  # update interval
    except KeyboardInterrupt:
        stop_video_counts()
