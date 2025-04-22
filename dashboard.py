from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import os
import sqlite3
import numpy as np
import face_recognition
from ultralytics import YOLO
from datetime import datetime
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
import threading

app = FastAPI()

# Directories for storing registered and unknown faces
registered_faces_dir = "registered_faces"
unknown_faces_dir = "unknown_faces"
os.makedirs(registered_faces_dir, exist_ok=True)
os.makedirs(unknown_faces_dir, exist_ok=True)

# SQLite database setup
db_file = "detection_records.db"
conn = sqlite3.connect(db_file, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    detected_object TEXT,
    confidence REAL,
    face_name TEXT
)
''')
conn.commit()

# Load YOLO model for weapon detection
weapon_model = YOLO('yolov5nu.pt')  # Ensure this model file exists in your project directory

# Load pre-trained face mask detection model
mask_net = cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')

# Generate anchors for face mask detection
anchors = generate_anchors(
    [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]],
    [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]],
    [[1, 0.62, 0.42]] * 5
)
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0, 0))

# Function to load registered faces
def load_registered_faces():
    encodings = []
    names = []
    for filename in os.listdir(registered_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(registered_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    names.append(os.path.splitext(filename)[0].rsplit("_", 1)[0])
    return encodings, names

known_face_encodings, known_face_names = load_registered_faces()

# Endpoint to register a new face
@app.post("/register-face/")
def register_face(name: str = Form(...)):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error: Could not open camera.")

    face_images = []

    while len(face_images) < 5:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to grab frame.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]
            face_images.append(face_image)

    cap.release()
    cv2.destroyAllWindows()

    if len(face_images) < 5:
        raise HTTPException(status_code=400, detail="Face registration failed. Please try again.")

    for i, face_image in enumerate(face_images):
        filepath = os.path.join(registered_faces_dir, f"{name}_{i}.jpg")
        cv2.imwrite(filepath, face_image)

    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_registered_faces()

    return JSONResponse(content={"message": f"Face registered successfully for {name}"})

# Function to log events into the database
def log_event(detected_object, confidence, face_name=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO detections (timestamp, detected_object, confidence, face_name)
    VALUES (?, ?, ?, ?)
    ''', (timestamp, detected_object, confidence, face_name))
    conn.commit()

# Detection loop in a separate thread
def detection_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = weapon_model(frame)
            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box
                    label = result.names[int(cls)]
                    confidence = float(conf)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    log_event(label, confidence)

            cv2.imshow('Home Security System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Endpoint to start the detection loop
@app.get("/start-detection/")
def start_detection():
    thread = threading.Thread(target=detection_loop)
    thread.start()
    return JSONResponse(content={"message": "Detection loop started. Press 'q' to quit the loop."})

# Endpoint to retrieve logged events
@app.get("/events/")
def get_events():
    cursor.execute("SELECT * FROM detections")
    rows = cursor.fetchall()
    return rows
