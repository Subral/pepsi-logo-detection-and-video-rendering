import cv2
import numpy as np
import base64
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS 
import threading

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

frame_base64 = None
logo_counter = 0
lock = threading.Lock()

sift = cv2.SIFT_create(
    nfeatures=4000,
    nOctaveLayers=3,
    contrastThreshold=0.06,
    edgeThreshold=7,
    sigma=1.6
)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

logo_image = cv2.imread('pepsi_logo.jpg', cv2.IMREAD_GRAYSCALE)

if logo_image is None:
    print("Error loading logo image")
    exit()

keypoints_logo, descriptors_logo = sift.detectAndCompute(logo_image, None)

cap = cv2.VideoCapture(0)

def encode_frame(frame):
    """Encode frame to JPEG format and then to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def track_object():
    global logo_counter, frame_base64

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        frame_base64 = encode_frame(frame)

        logo_detected = False

        if des2 is not None:
            matches = flann.knnMatch(descriptors_logo, des2, k=2)

            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > 40:
                logo_detected = True
                logo_counter += 1
                print(f"Pepsi logo detected {logo_counter} times.")
                
                # Emit object_detected event when the logo is detected
                socketio.emit('object_detected', {'detected': True})

        if not logo_detected:
            # Emit object_detected event with detected: False
            socketio.emit('object_detected', {'detected': False})

        with lock:
            socketio.emit('update_frame', {'frame': frame_base64})
            socketio.emit('update_logo_counter', {'counter': logo_counter})

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracking_thread = threading.Thread(target=track_object)
    tracking_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)
