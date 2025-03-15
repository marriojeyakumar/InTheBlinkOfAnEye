import cv2
import dlib
import time
from imutils import face_utils
from scipy.spatial import distance as dist
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize dlib's face detector and load the facial landmark predictor.
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this file is in your project root
predictor = dlib.shape_predictor(predictor_path)

# Open the default webcam.
cap = cv2.VideoCapture(0)

# Blink detection parameters.
BLINK_THRESHOLD = 0.2         # Eye Aspect Ratio threshold.
CONSECUTIVE_FRAMES = 2        # Number of consecutive frames the eye must be below threshold.
blink_counter = 0
blink_cooldown = 2.0          # Minimum seconds between captures.
last_blink_time = time.time()

def eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio (EAR) for a given set of eye landmarks.
    The formula is: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame():
    global blink_counter, last_blink_time
    ret, frame = cap.read()
    if not ret:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the coordinates for the left and right eyes.
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute EAR for both eyes and take the average.
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Optional: draw contours around the eyes for visualization.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Blink detection logic.
        if ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSECUTIVE_FRAMES:
                # Ensure a cooldown so that we don't capture too frequently.
                if time.time() - last_blink_time > blink_cooldown:
                    cv2.putText(frame, "Blink Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    # Save the current frame as the captured image.
                    cv2.imwrite("static/captured.jpg", frame)
                    last_blink_time = time.time()
            blink_counter = 0

    return frame

def generate_frames():
    """
    A generator function that captures frames from the webcam,
    processes them, and yields them as JPEG-encoded byte arrays for streaming.
    """
    while True:
        frame = process_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield the frame in the proper multipart format.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Render the main HTML page.
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type).
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
