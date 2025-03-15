import cv2
import dlib
import time
from datetime import datetime, timedelta
from imutils import face_utils
from scipy.spatial import distance as dist
from flask import Flask, render_template, Response
from collections import deque


# ------------------------- Constants -------------------------
# Blink detection parameters
EYE_AR_THRESH = 0.2            # Eye Aspect Ratio below which the eye is considered closed.
ATTACK_FRAMES_TOTAL = 3        # Total closed frames required.
ATTACK_FRAMES_CONSECUTIVE = 2   # Consecutive closed frames required.
RELEASE_FRAMES = 5             # Number of consecutive open frames to reset counters.
DELAY_TIME = 2.0               # Minimum seconds between captures.
DELTA_IMMEDIATE = 0.1          # Immediate capture delta for EAR history.

# For EAR history (to detect rapid drop in EAR)
MAX_HISTORY_LENGTH = 8
frame_buffer = deque(maxlen=30)
# ------------------------- Initialization -------------------------
# Initialize Flask
app = Flask(__name__)

# Initialize dlib's face detector and load the 68-point facial landmark predictor.
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Open the default webcam with higher resolution.
print("[INFO] Starting video capture...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Global variables for blink detection logic.
last_photo_time = datetime.now() - timedelta(seconds=DELAY_TIME)
total_closed = 0
consecutive_closed = 0
consecutive_open = 0
ear_history = []

# ------------------------- Helper Functions -------------------------
def eye_aspect_ratio(eye):
    """
    Compute the Eye Aspect Ratio (EAR) for a given eye.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def process_frame():
    global total_closed, consecutive_closed, consecutive_open, last_photo_time, ear_history, frame_buffer

    ret, frame_raw = cap.read()
    if not ret:
        return None

    # Resize frame for processing (keeping a copy of the raw frame for snapshot)
    frame = cv2.resize(frame_raw, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rects = detector(gray, 0)
    ear = None

    if rects:
        ears = []
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            current_ear = (leftEAR + rightEAR) / 2.0
            ears.append(current_ear)

            # Draw contours around the eyes for visualization.
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        ear = sum(ears) / len(ears)
        ear_history.append(ear)
        if len(ear_history) > MAX_HISTORY_LENGTH:
            ear_history.pop(0)
    else:
        # If no face is detected, push a high EAR value (or skip buffering)
        ear = 1.0

    # Add the raw frame and its EAR to the buffer
    frame_buffer.append((frame_raw.copy(), ear))

    # Update blink counters based on EAR threshold.
    if ear < EYE_AR_THRESH:
        total_closed += 1
        consecutive_closed += 1
        consecutive_open = 0
    else:
        consecutive_closed = 0
        consecutive_open += 1

    if consecutive_open > RELEASE_FRAMES:
        total_closed = 0

    immediate_capture = False
    if len(ear_history) == MAX_HISTORY_LENGTH:
        avg_ear = sum(ear_history) / MAX_HISTORY_LENGTH
        min_ear = min(ear_history)
        if (avg_ear - min_ear) > DELTA_IMMEDIATE:
            immediate_capture = True

    should_capture = (((total_closed >= ATTACK_FRAMES_TOTAL and consecutive_closed >= ATTACK_FRAMES_CONSECUTIVE) 
                       or immediate_capture)
                       and ((datetime.now() - last_photo_time).total_seconds() > DELAY_TIME))

    if should_capture and frame_buffer:
        cv2.putText(frame, "Blink Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Select the frame from the buffer with the minimum EAR
        best_frame, best_ear = min(frame_buffer, key=lambda x: x[1])
        cv2.imwrite("static/captured.jpg", best_frame)
        last_photo_time = datetime.now()
        total_closed = 0
        consecutive_closed = 0
        consecutive_open = 0
        # Clear the buffer after capture if desired
        frame_buffer.clear()

    cv2.putText(frame, f"Faces: {len(rects)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if ear is not None:
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame

def generate_frames():
    """
    Generator function that captures processed frames and yields
    them as JPEG-encoded byte arrays for streaming.
    """
    while True:
        frame = process_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ------------------------- Flask Routes -------------------------
@app.route('/')
def index():
    # Render the main page.
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Stream the video feed.
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------- Main Entry Point -------------------------
if __name__ == '__main__':
    app.run(debug=True)
