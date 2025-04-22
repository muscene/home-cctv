import cv2
import os
import sqlite3
import numpy as np
import face_recognition
from ultralytics import YOLO
from paddle.inference import Config, create_predictor
from datetime import datetime
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from config import *

# Initialize SQLite database
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
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

# Weapon Detection Model
weapon_model = YOLO(WEAPON_MODEL_PATH)

# Mask Detection Model
def load_mask_predictor(model_dir):
    config = Config(os.path.join(model_dir, "__model__"), os.path.join(model_dir, "__params__"))
    config.disable_gpu()
    config.enable_memory_optim()
    return create_predictor(config)

mask_predictor = load_mask_predictor(MASK_MODEL_DIR)
anchors = generate_anchors([[33, 33], [17, 17], [9, 9]], [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22]], [[1, 0.62, 0.42]] * 3)
anchors_exp = np.expand_dims(anchors, axis=0)

# Email Alert Function
def send_email(subject, message_body, image_path=None):
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    import smtplib

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = subject

    msg.attach(MIMEText(message_body, 'plain'))

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read(), name=os.path.basename(image_path))
            msg.attach(img)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
    except Exception as e:
        print(f"Error sending email: {e}")

# Log Events in the Database
def log_event(detected_object, confidence, face_name=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO detections (timestamp, detected_object, confidence, face_name)
    VALUES (?, ?, ?, ?)
    ''', (timestamp, detected_object, confidence, face_name))
    conn.commit()

# Face Recognition Initialization
def load_registered_faces():
    encodings, names = [], []
    for filename in os.listdir(REGISTERED_FACES_DIR):
        filepath = os.path.join(REGISTERED_FACES_DIR, filename)
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            face_encodings = face_recognition.face_encodings(image, face_locations)
            encodings.append(face_encodings[0])
            names.append(os.path.splitext(filename)[0])
    return encodings, names

known_face_encodings, known_face_names = load_registered_faces()
