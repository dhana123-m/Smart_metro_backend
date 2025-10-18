from flask import Flask, jsonify, request
from flask_cors import CORS
from detection.people_counter import (
    start_video_counts,
    stop_video_counts,
    get_latest_counts,
    get_occupancy
)

app = Flask(__name__)
CORS(app)  # enable CORS for all routes

@app.route("/api/start")
def start_api():
    """
    Example request:
    /api/start?comp1=data/videos/comp1.mp4&comp2=data/videos/comp2.mp4
    """
    sources = {}
    for key, value in request.args.items():
        sources[f"compartment_{key}"] = value if value != "webcam" else 0

    if not sources:
        return jsonify({"error": "No video sources provided"}), 400

    start_video_counts(sources)
    return jsonify({"message": f"Started processing {len(sources)} compartments"})

@app.route("/api/stop")
def stop_api():
    stop_video_counts()
    return jsonify({"message": "Video processing stopped"})

@app.route("/api/occupancy")
def occupancy_api():
    counts = get_latest_counts()
    coach_capacity = 120
    result = get_occupancy(counts, coach_capacity)
    return jsonify(result)

if __name__ == "__main__":
    # Allow external devices (like your phone) to connect
    app.run(host="0.0.0.0", port=5000, debug=True)
