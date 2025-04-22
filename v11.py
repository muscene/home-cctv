import cv2
import numpy as np
import os
import time
import threading
import json
import uuid
import datetime
from flask import Flask, render_template, Response, request, jsonify, flash, redirect, url_for
from queue import Queue
from deepface import DeepFace
import base64
import h5py
import secrets
import logging
import psutil
import gc
from collections import defaultdict
from sklearn.cluster import DBSCAN
import shutil
from datetime import timedelta
import weakref
import csv
from io import StringIO
import torch
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Configure logging
logging.basicConfig(
    filename='static/logs/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Create necessary directories
os.makedirs('static/captures', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('static/faces/samples', exist_ok=True)
os.makedirs('static/logs', exist_ok=True)
os.makedirs('static/models', exist_ok=True)

# Camera setup
camera = None
camera_active = True
output_frame = None
frame_lock = threading.Lock()
face_detection_queue = Queue(maxsize=50)  # Increased for Raspberry Pi
object_detection_queue = Queue(maxsize=50)

# Face recognition variables
known_face_encodings = []
known_face_names = []
known_face_access = {}
face_detection_active = True
last_face_scan_time = 0
face_scan_interval = 2.0  # Increased for Raspberry Pi
face_recognition_model = 'ArcFace'
face_confidence_threshold = 0.6
last_face_locations = []
last_face_names = []

# Object detection variables
object_detection_active = True
last_object_scan_time = 0
object_scan_interval = 3.0  # Increased for Raspberry Pi
object_confidence_threshold = 0.5
target_object_classes = ['person', 'car', 'dog', 'knife', 'gun']
yolo_model = None

# Mask detection variables
mask_detection_active = True
last_mask_scan_time = 0
mask_scan_interval = 2.0  # Increased for Raspberry Pi
mask_confidence_threshold = 0.5
mask_net = None

# Security logs
security_logs = []

# Continuous learning parameters
continuous_learning = True
learning_threshold = 0.7
max_samples_per_face = 10
learning_rate = 0.3
min_learning_interval = 3600
last_face_update = {}

# Memory monitoring
MEMORY_THRESHOLD = 1_000_000_000

# Email rate-limiting
last_email_times = defaultdict(float)
email_cooldown = 60  # seconds

# DNN face detector with error handling
prototxt_path = 'static/models/deploy.prototxt'
caffemodel_path = 'static/models/res10_300x300_ssd_iter_140000.caffemodel'
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
except FileNotFoundError:
    logging.error(f"DNN model files missing: {prototxt_path} or {caffemodel_path}")
    net = None
    face_detection_active = False

# Mask detector with error handling
mask_prototxt_path = 'static/models/face_mask_detection.prototxt'
mask_caffemodel_path = 'static/models/face_mask_detection.caffemodel'
try:
    mask_net = cv2.dnn.readNetFromCaffe(mask_prototxt_path, mask_caffemodel_path)
except FileNotFoundError:
    logging.error(f"Mask detection model files missing: {mask_prototxt_path} or {mask_caffemodel_path}")
    mask_net = None
    mask_detection_active = False

# Preload YOLO model
try:
    yolo_model = YOLO('static/models/yolov5s.pt')
    logging.info("Preloaded YOLOv5 model")
except FileNotFoundError:
    logging.error("YOLO model file missing: static/models/yolov5s.pt")
    yolo_model = None
    object_detection_active = False

def monitor_memory():
    """Monitor memory usage and trigger garbage collection if needed."""
    process = psutil.Process()
    mem_info = process.memory_info()
    if mem_info.rss > MEMORY_THRESHOLD:
        logging.warning(f"High memory usage: {mem_info.rss} bytes")
        gc.collect()

def save_face_encodings():
    """Save face encodings to HDF5 file with metadata."""
    try:
        with h5py.File('static/faces/encodings.h5', 'w') as f:
            f.create_dataset('names', data=np.array(known_face_names, dtype='S'))
            f.create_dataset('encodings', data=np.array(known_face_encodings))
            access_names = list(known_face_access.keys())
            access_levels = list(known_face_access.values())
            f.create_dataset('access_names', data=np.array(access_names, dtype='S'))
            f.create_dataset('access_levels', data=np.array(access_levels, dtype='S'))
        logging.info(f"Saved {len(known_face_names)} face encodings to HDF5")
    except Exception as e:
        logging.error(f"Error saving face encodings: {e}")

def load_face_encodings():
    """Load face encodings from HDF5 file."""
    global known_face_encodings, known_face_names, known_face_access
    try:
        if os.path.exists('static/faces/encodings.h5'):
            with h5py.File('static/faces/encodings.h5', 'r') as f:
                known_face_names = [name.decode('utf-8') for name in f['names'][:]]
                known_face_encodings = f['encodings'][:].tolist()
                access_names = [name.decode('utf-8') for name in f['access_names'][:]]
                access_levels = [level.decode('utf-8') for level in f['access_levels'][:]]
                known_face_access = dict(zip(access_names, access_levels))
            logging.info(f"Loaded {len(known_face_names)} face encodings from HDF5")
            return True
    except Exception as e:
        logging.error(f"Error loading face encodings: {e}")
    return False

def cluster_encodings(name, encodings):
    """Cluster face encodings to group similar samples."""
    if len(encodings) < 2:
        return encodings
    try:
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(encodings)
        core_samples = np.where(clustering.labels_ != -1)[0]
        if len(core_samples) > 0:
            return [encodings[i] for i in core_samples]
        return encodings
    except Exception as e:
        logging.error(f"Error clustering encodings for {name}: {e}")
        return encodings

def update_face_encoding(name, new_encoding, confidence, face_location, frame):
    """Update face encoding using a weighted average."""
    global known_face_encodings, known_face_names, last_face_update
    current_time = time.time()
    
    top, right, bottom, left = face_location
    face_size = (bottom - top) * (right - left)
    if face_size < 10000:
        logging.warning(f"Sample for {name} rejected: insufficient quality")
        return False
    
    if name in last_face_update and current_time - last_face_update[name] < min_learning_interval:
        return False
    
    if name in known_face_names:
        idx = known_face_names.index(name)
        current_encoding = known_face_encodings[idx]
        samples_path = f'static/faces/samples/{name.replace(" ", "_")}.h5'
        
        samples = []
        try:
            if os.path.exists(samples_path):
                with h5py.File(samples_path, 'r') as f:
                    samples = f['samples'][:].tolist()
        except Exception as e:
            logging.error(f"Error loading samples for {name}: {e}")
        
        samples.append(new_encoding)
        samples = cluster_encodings(name, samples)
        
        if len(samples) > max_samples_per_face:
            samples = samples[-max_samples_per_face:]
            logging.info(f"Trimmed samples for {name} to {max_samples_per_face}")
        
        try:
            with h5py.File(samples_path, 'w') as f:
                f.create_dataset('samples', data=np.array(samples))
        except Exception as e:
            logging.error(f"Error saving samples for {name}: {e}")
        
        updated_encoding = np.mean(samples, axis=0)
        known_face_encodings[idx] = updated_encoding
        last_face_update[name] = current_time
        save_face_encodings()
        logging.info(f"Updated face encoding for {name} with {len(samples)} samples")
        return True
    
    return False

def load_known_faces():
    """Load known faces from the faces directory."""
    global known_face_encodings, known_face_names, known_face_access
    if load_face_encodings():
        return
    
    known_face_encodings = []
    known_face_names = []
    known_face_access = {}
    
    try:
        face_db_path = 'static/faces/face_db.json'
        face_db = json.load(open(face_db_path, 'r')) if os.path.exists(face_db_path) else {}
        
        for filename in os.listdir('static/faces'):
            if not filename.endswith(('.jpg', '.png')) or not filename.startswith('face_'):
                continue
            face_id = os.path.splitext(filename)[0]
            face_image_path = os.path.join('static/faces', filename)
            
            try:
                face_image = cv2.imread(face_image_path)
                embedding = DeepFace.represent(face_image, model_name=face_recognition_model, enforce_detection=False)
                if embedding:
                    name = face_db.get(face_id, {}).get('name', f"Person {len(known_face_names) + 1}")
                    access_level = face_db.get(face_id, {}).get('access_level', 'limited')
                    known_face_encodings.append(embedding[0]['embedding'])
                    known_face_names.append(name)
                    known_face_access[name] = access_level
                    logging.info(f"Loaded face: {name} with access level: {access_level}")
                else:
                    logging.warning(f"No face encoding generated for {filename}")
            except Exception as e:
                logging.error(f"Error processing face image {filename}: {e}")
        
        save_face_encodings()
        logging.info(f"Loaded {len(known_face_names)} known faces")
    except Exception as e:
        logging.error(f"Error loading known faces: {e}")

def load_security_logs():
    """Load security logs from file."""
    global security_logs
    try:
        with open('static/logs/security_logs.json', 'r') as f:
            security_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading security logs: {e}")
        security_logs = []

def save_security_logs():
    """Save security logs to file."""
    try:
        with open('static/logs/security_logs.json', 'w') as f:
            json.dump(security_logs, f)
    except Exception as e:
        logging.error(f"Error saving security logs: {e}")

def get_face_description(name):
    """Generate a detailed description for a recognized face."""
    try:
        face_db_path = 'static/faces/face_db.json'
        face_db = json.load(open(face_db_path, 'r')) if os.path.exists(face_db_path) else {}
        
        for face_id, data in face_db.items():
            if data.get('name') == name:
                return (
                    f"Name: {name}, "
                    f"Access Level: {data.get('access_level', 'Unknown')}, "
                    f"Relationship: {data.get('relationship', 'None')}, "
                    f"Notes: {data.get('notes', 'None')}, "
                    f"Registered At: {data.get('registered_at', 'Unknown')}"
                )
        return f"Name: {name}, Access Level: {known_face_access.get(name, 'Unknown')}, No further details available"
    except Exception as e:
        logging.error(f"Error generating face description for {name}: {e}")
        return f"Name: {name}, Description unavailable"

def add_security_log(event, person=None, confidence=0.0, status="Unknown", description=None, metadata=None):
    """Add a security log entry with face or object description."""
    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "person": person,
        "confidence": float(confidence),
        "status": status,
        "description": description or (get_face_description(person) if person and person != "Unknown" else "No description available"),
        "metadata": metadata or {}
    }
    security_logs.insert(0, log_entry)
    if len(security_logs) > 1000:
        security_logs.pop()
    save_security_logs()
    logging.info(f"Added security log: {event} for {person or 'Unknown'}")
    return log_entry

def send_email(subject, body, attachments=None, event_type="general"):
    """Send an email with optional attachments, with rate-limiting."""
    try:
        current_time = time.time()
        if current_time - last_email_times[event_type] < email_cooldown:
            logging.warning(f"Email for {event_type} skipped due to cooldown")
            return False
        
        email_address = os.environ.get('EMAIL_ADDRESS')
        email_password = os.environ.get('EMAIL_PASSWORD')
        email_recipient = os.environ.get('EMAIL_RECIPIENT')
        
        if not all([email_address, email_password, email_recipient]):
            logging.error("Email configuration missing")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = email_recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        if attachments:
            for file_path in attachments:
                if not os.path.exists(file_path):
                    logging.error(f"Attachment not found: {file_path}")
                    continue
                with open(file_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                filename = os.path.basename(file_path)
                part.add_header('Content-Disposition', f'attachment; filename={filename}')
                msg.attach(part)
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_address, email_password)
            server.send_message(msg)
        
        last_email_times[event_type] = current_time
        logging.info(f"Email sent for {event_type}: {subject}")
        return True
    except Exception as e:
        logging.error(f"Error sending email for {event_type}: {e}")
        return False

def record_video(frames, duration=3, fps=10):
    """Record a video from frames for the specified duration."""
    try:
        timestamp = int(time.time())
        video_path = f"static/captures/video_{timestamp}.avi"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in frames[:int(duration * fps)]:
            out.write(frame)
        
        out.release()
        logging.info(f"Video recorded: {video_path}")
        return video_path
    except Exception as e:
        logging.error(f"Error recording video: {e}")
        return None

def capture_photos(frames, count=3, prefix="capture"):
    """Capture a specified number of photos from frames."""
    try:
        timestamp = int(time.time())
        photo_paths = []
        for i in range(min(count, len(frames))):
            photo_path = f"static/captures/{prefix}_{timestamp}_{i}.jpg"
            cv2.imwrite(photo_path, frames[i])
            photo_paths.append(photo_path)
            logging.info(f"Photo captured: {photo_path}")
        return photo_paths
    except Exception as e:
        logging.error(f"Error capturing photos: {e}")
        return []

def get_camera(max_retries=3, backoff_factor=2):
    """Initialize and return the camera with retry logic."""
    global camera
    if camera is None or not camera.isOpened():
        retries = 0
        camera_source = os.environ.get('CAMERA_SOURCE', '0')
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass
        
        while retries < max_retries:
            try:
                logging.info(f"Connecting to camera: {camera_source}")
                camera = cv2.VideoCapture(camera_source)
                if camera.isOpened():
                    if isinstance(camera_source, int):
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced for Raspberry Pi
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    logging.info("Camera connected successfully")
                    return camera
                else:
                    raise cv2.error("Camera not opened")
            except Exception as e:
                logging.error(f"Camera connection attempt {retries + 1} failed: {e}")
                time.sleep(backoff_factor ** retries)
                retries += 1
        logging.error(f"Failed to connect to camera after {max_retries} attempts")
        return None
    return camera

def detect_mask(frame, face_locations):
    """Detect if faces are wearing masks and capture photos for masked faces."""
    global last_mask_scan_time
    current_time = time.time()
    if not mask_detection_active or (current_time - last_mask_scan_time < mask_scan_interval) or mask_net is None:
        return []
    
    last_mask_scan_time = current_time
    
    try:
        mask_statuses = []
        height, width = frame.shape[:2]
        
        frames_for_photos = [frame.copy()]
        for _ in range(2):  # Capture 3 frames total
            success, next_frame = camera.read()
            if success:
                frames_for_photos.append(next_frame.copy())
        
        for (top, right, bottom, left) in face_locations:
            face_crop = frame[max(0, top):min(height, bottom), max(0, left):min(width, right)]
            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (104.0, 177.0, 123.0))
            mask_net.setInput(blob)
            detections = mask_net.forward()
            
            mask_confidence = detections[0, 1]
            no_mask_confidence = detections[0, 0]
            is_masked = mask_confidence > no_mask_confidence and mask_confidence > mask_confidence_threshold
            mask_statuses.append('Masked' if is_masked else 'No Mask')
            
            if is_masked:
                # Capture photos for masked face
                photo_paths = capture_photos(frames_for_photos, count=3, prefix="masked_face")
                if photo_paths:
                    threading.Thread(target=send_email, args=(
                        "Security Alert: Masked Face Detected",
                        f"A masked face was detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with confidence {mask_confidence:.2f}. Photos attached.",
                        photo_paths,
                        "masked_face"
                    )).start()
                    add_security_log(
                        "Masked Face Detected", 
                        None, 
                        mask_confidence, 
                        "Warning",
                        description=f"Masked face detected with confidence {mask_confidence:.2f}",
                        metadata={'photo_paths': photo_paths}
                    )
        
        return mask_statuses
    except Exception as e:
        logging.error(f"Mask detection error: {e}")
        return []
def detect_faces(frame, small_frame):
    """Detect and recognize faces using DNN and DeepFace, with mask detection but no photo capture for unknown faces."""
    global last_face_scan_time, last_face_locations, last_face_names
    current_time = time.time()
    if not face_detection_active or (current_time - last_face_scan_time < face_scan_interval) or net is None:
        return last_face_locations, last_face_names, [], []
    
    last_face_scan_time = current_time
    
    try:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        face_locations = []
        height, width = frame.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (left, top, right, bottom) = box.astype("int")
                if left < 0 or top < 0 or right > width or bottom > height:
                    continue
                face_locations.append((top, right, bottom, left))
        
        face_names = []
        face_statuses = []
        mask_statuses = detect_mask(frame, face_locations)
        
        # Temporal coherence: Skip recognition if faces haven't moved significantly
        if last_face_locations and len(face_locations) == len(last_face_locations):
            iou_threshold = 0.8
            skip_recognition = True
            for (t1, r1, b1, l1), (t2, r2, b2, l2) in zip(face_locations, last_face_locations):
                x_left = max(l1, l2)
                y_top = max(t1, t2)
                x_right = min(r1, r2)
                y_bottom = min(b1, b2)
                if x_right <= x_left or y_bottom <= y_top:
                    skip_recognition = False
                    break
                intersection = (x_right - x_left) * (y_bottom - y_top)
                union = ((r1 - l1) * (b1 - t1)) + ((r2 - l2) * (b2 - t2)) - intersection
                iou = intersection / union
                if iou < iou_threshold:
                    skip_recognition = False
                    break
            if skip_recognition:
                return last_face_locations, last_face_names, [known_face_access.get(name, 'Unauthorized') for name in last_face_names], mask_statuses
        
        for face_location, mask_status in zip(face_locations, mask_statuses):
            top, right, bottom, left = face_location
            face_crop = frame[max(0, top):min(height, bottom), max(0, left):min(width, right)]
            
            name = "Unknown"
            status = "Unauthorized"
            confidence = 0.0
            metadata = {}
            
            try:
                embedding = DeepFace.represent(face_crop, model_name=face_recognition_model, enforce_detection=False)
                if embedding:
                    embedding = embedding[0]['embedding']
                    if known_face_encodings:
                        distances = DeepFace.verify(
                            [embedding] * len(known_face_encodings),
                            known_face_encodings,
                            model_name=face_recognition_model,
                            distance_metric='cosine'
                        )
                        min_distance = float('inf')
                        best_match_idx = -1
                        for idx, result in enumerate(distances):
                            dist = result['distance']
                            if dist < min_distance and dist < face_confidence_threshold:
                                min_distance = dist
                                best_match_idx = idx
                        
                        if best_match_idx >= 0:
                            name = known_face_names[best_match_idx]
                            confidence = 1.0 - min_distance
                            if known_face_access.get(name) in ['full', 'limited']:
                                status = "Authorized"
                            if continuous_learning and confidence > learning_threshold:
                                update_face_encoding(name, embedding, confidence, face_location, frame)
                            metadata['match_distance'] = min_distance
                    else:
                        # No known faces registered, log unknown but don't save
                        add_security_log("Unknown Face", None, 0.0, "Unauthorized")
            except Exception as e:
                logging.error(f"Face recognition error: {e}")
            
            # Skip photo capture and email for unknown faces
            if name == "Unknown":
                face_names.append(name)
                face_statuses.append(status)
                continue
            
            # Log recognized faces only
            face_names.append(name)
            face_statuses.append(status)
            add_security_log("Face Detection", name, confidence, status, metadata=metadata)
        
        last_face_locations = face_locations
        last_face_names = face_names
        return face_locations, face_names, face_statuses, mask_statuses
    
    except Exception as e:
        logging.error(f"Face detection error: {e}")
        return [], [], [], []
def detect_objects(frame):
    """Detect objects in the frame using YOLOv5, with photo capture for weapons."""
    global last_object_scan_time, yolo_model
    current_time = time.time()
    if not object_detection_active or (current_time - last_object_scan_time < object_scan_interval) or yolo_model is None:
        return [], []
    
    last_object_scan_time = current_time
    
    try:
        results = yolo_model(frame, conf=object_confidence_threshold)
        object_locations = []
        object_names = []
        
        frames_for_photos = [frame.copy()]
        for _ in range(2):  # Capture 3 frames total
            success, next_frame = camera.read()
            if success:
                frames_for_photos.append(next_frame.copy())
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = yolo_model.names[class_id]
                if class_name in target_object_classes or not target_object_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    object_locations.append((y1, x2, y2, x1))
                    object_names.append(class_name)
                    status = "Warning" if class_name in ['knife', 'gun'] else "Detected"
                    metadata = {}
                    
                    if class_name in ['knife', 'gun']:
                        photo_paths = capture_photos(frames_for_photos, count=3, prefix=f"{class_name}")
                        if photo_paths:
                            threading.Thread(target=send_email, args=(
                                f"Security Alert: {class_name.capitalize()} Detected",
                                f"A {class_name} was detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with confidence {confidence:.2f}. Photos attached.",
                                photo_paths,
                                "weapon"
                            )).start()
                            metadata['photo_paths'] = photo_paths
                    
                    add_security_log(
                        f"Object Detection: {class_name}", 
                        None, 
                        confidence, 
                        status,
                        description=f"Object: {class_name}, Confidence: {confidence:.2f}", 
                        metadata=metadata
                    )
        
        return object_locations, object_names
    
    except Exception as e:
        logging.error(f"Object detection error: {e}")
        return [], []

def process_detections():
    """Process frames from the queues for face and object detection."""
    global output_frame, frame_lock, camera_active
    while True:
        if not camera_active:
            time.sleep(1)
            continue
        
        try:
            frame = None
            annotated_frame = None
            small_frame = None
            
            if not face_detection_queue.empty():
                frame = face_detection_queue.get()
            elif not object_detection_queue.empty():
                frame = object_detection_queue.get()
            else:
                time.sleep(0.01)
                continue
            
            annotated_frame = frame.copy()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Face detection
            face_locations, face_names, face_statuses, mask_statuses = detect_faces(frame, small_frame)
            for (top, right, bottom, left), name, status, mask_status in zip(face_locations, face_names, face_statuses, mask_statuses):
                color = (0, 255, 0) if status == "Authorized" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(annotated_frame, f"{name} ({mask_status})", (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, status, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if name != "Unknown":
                    description = get_face_description(name)
                    for i, line in enumerate(description.split(', ')):
                        cv2.putText(annotated_frame, line, (left, top - 40 - i*20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Object detection
            object_locations, object_names = detect_objects(frame)
            for (top, right, bottom, left), name in zip(object_locations, object_names):
                color = (255, 0, 0) if name in ['knife', 'gun'] else (0, 255, 255)
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(annotated_frame, name, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            status_text = (
                f"Face Recognition: {'Active' if face_detection_active else 'Inactive'}, "
                f"Object Detection: {'Active' if object_detection_active else 'Inactive'}, "
                f"Mask Detection: {'Active' if mask_detection_active else 'Inactive'}"
            )
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            with frame_lock:
                output_frame = annotated_frame.copy()
            
            frame = safe_release(frame)
            annotated_frame = safe_release(annotated_frame)
            small_frame = safe_release(small_frame)
        
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            time.sleep(0.1)

def generate_frames():
    """Generate frames for the video feed."""
    global output_frame, frame_lock, camera_active
    while True:
        frame = None
        display_frame = None
        blank_frame = None
        buffer = None
        
        try:
            camera = get_camera()
            if camera is None or not camera_active:
                blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                message = "Camera Disconnected" if camera_active else "Camera Not Available"
                cv2.putText(blank_frame, message, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1.0)
                continue
            
            success, frame = camera.read()
            if not success:
                logging.warning("Failed to read frame, attempting reconnect")
                camera.release()
                camera = None
                time.sleep(0.5)
                continue
            
            if not face_detection_queue.full() and face_detection_active:
                face_detection_queue.put(frame.copy())
            if not object_detection_queue.full() and object_detection_active:
                object_detection_queue.put(frame.copy())
            if face_detection_queue.full() or object_detection_queue.full():
                logging.warning("Detection queue full, dropping frame")
                add_security_log("Queue Full", None, 0.0, "Warning")
            
            with frame_lock:
                display_frame = output_frame.copy() if output_frame is not None else frame.copy()
            
            _, buffer = cv2.imencode('.jpg', display_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
            time.sleep(0.1)
        finally:
            frame = safe_release(frame)
            display_frame = safe_release(display_frame)
            blank_frame = safe_release(blank_frame)
            buffer = safe_release(buffer)
            monitor_memory()

@app.route('/')
def index():
    """Video streaming home page."""
    camera_status = "Connected" if camera_active else "Disconnected"
    return render_template('index.html', camera_status=camera_status)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/configure_camera', methods=['GET', 'POST'])
def configure_camera():
    """Configure camera source."""
    global camera, camera_active
    if request.method == 'POST':
        try:
            camera_source = request.form.get('camera_source')
            if camera is not None:
                camera.release()
                camera = None
            os.environ['CAMERA_SOURCE'] = camera_source
            camera = get_camera()
            if camera and camera.isOpened():
                camera_active = True
                flash("Camera connected successfully!", "success")
            else:
                flash("Failed to connect to camera.", "error")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Error configuring camera: {e}")
            flash(f"Error configuring camera: {str(e)}", "error")
    
    current_source = os.environ.get('CAMERA_SOURCE', '0')
    return render_template('configure_camera.html', current_source=current_source)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle face registration."""
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            relationship = request.form.get('relationship')
            access_level = request.form.get('access_level')
            notes = request.form.get('notes')
            face_image = request.form.get('face_image')
            
            if not name or not access_level:
                flash("Name and access level are required.", "error")
                return render_template('register.html')
            
            face_id = f"face_{uuid.uuid4()}"
            image_path = f'static/faces/{face_id}.jpg'
            frame_to_save = None
            face_img = None
            
            if face_image and face_image.startswith('data:image'):
                image_data = face_image.split(',')[1]
                if len(image_data) > 10_000_000:
                    flash("Image too large (max 10MB).", "error")
                    return render_template('register.html')
                image_bytes = base64.b64decode(image_data)
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                logging.info(f"Saved face image from base64 to {image_path}")
            else:
                with frame_lock:
                    if output_frame is not None:
                        frame_to_save = output_frame.copy()
                    else:
                        flash("No camera frame available.", "error")
                        return render_template('register.html')
                cv2.imwrite(image_path, frame_to_save)
                logging.info(f"Saved current frame to {image_path}")
                frame_to_save = None
            
            face_img = cv2.imread(image_path)
            embedding = DeepFace.represent(face_img, model_name=face_recognition_model, enforce_detection=False)
            if not embedding:
                flash("No face detected in the image.", "error")
                return render_template('register.html')
            
            face_db_path = 'static/faces/face_db.json'
            face_db = json.load(open(face_db_path, 'r')) if os.path.exists(face_db_path) else {}
            face_db[face_id] = {
                'name': name,
                'relationship': relationship,
                'access_level': access_level,
                'notes': notes,
                'registered_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(face_db_path, 'w') as f:
                json.dump(face_db, f)
            
            load_known_faces()
            add_security_log("Face Registration", name, 1.0, "System")
            flash(f"Face registered successfully for {name}!", "success")
            return redirect(url_for('index'))
        
        except Exception as e:
            logging.error(f"Registration error: {e}")
            flash(f"Error during registration: {str(e)}", "error")
    
    return render_template('register.html')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture a frame for face registration."""
    try:
        timestamp = int(time.time())
        image_path = f"static/captures/capture_{timestamp}.jpg"
        
        with frame_lock:
            if output_frame is not None:
                frame_to_save = output_frame.copy()
            else:
                return jsonify({"success": False, "error": "No camera frame available"})
        
        cv2.imwrite(image_path, frame_to_save)
        frame_to_save = None
        
        face_img = cv2.imread(image_path)
        embedding = DeepFace.represent(face_img, model_name=face_recognition_model, enforce_detection=False)
        face_detected = bool(embedding)
        warning_message = "No face detected in the captured frame." if not face_detected else None
        
        face_img = None
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        result = {
            "success": True,
            "image_data": f"data:image/jpeg;base64,{encoded_image}",
            "image_path": image_path
        }
        if warning_message:
            result["warning"] = warning_message
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error in capture_frame: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/toggle_face_detection', methods=['POST'])
def toggle_face_detection():
    """Toggle face detection on/off."""
    global face_detection_active
    try:
        data = request.json
        action = data.get('action', 'toggle')
        face_detection_active = (action == 'enable') or (action == 'toggle' and not face_detection_active)
        return jsonify({"success": True, "face_detection_active": face_detection_active})
    except Exception as e:
        logging.error(f"Error toggling face detection: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/toggle_object_detection', methods=['POST'])
def toggle_object_detection():
    """Toggle object detection on/off."""
    global object_detection_active
    try:
        data = request.json
        action = data.get('action', 'toggle')
        object_detection_active = (action == 'enable') or (action == 'toggle' and not object_detection_active)
        return jsonify({"success": True, "object_detection_active": object_detection_active})
    except Exception as e:
        logging.error(f"Error toggling object detection: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/toggle_mask_detection', methods=['POST'])
def toggle_mask_detection():
    """Toggle mask detection on/off."""
    global mask_detection_active
    try:
        data = request.json
        action = data.get('action', 'toggle')
        mask_detection_active = (action == 'enable') or (action == 'toggle' and not mask_detection_active)
        return jsonify({"success": True, "mask_detection_active": mask_detection_active})
    except Exception as e:
        logging.error(f"Error toggling mask detection: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/toggle_learning', methods=['POST'])
def toggle_learning():
    """Toggle continuous learning on/off."""
    global continuous_learning
    try:
        data = request.json
        action = data.get('action', 'toggle')
        continuous_learning = (action == 'enable') or (action == 'toggle' and not continuous_learning)
        return jsonify({"success": True, "continuous_learning": continuous_learning})
    except Exception as e:
        logging.error(f"Error toggling learning: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/logs')
def logs_page():
    """Display security logs."""
    if request.args.get('format') == 'json':
        limit = request.args.get('limit', 5, type=int)
        return jsonify({"logs": security_logs[:limit]})
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    filtered_logs = security_logs
    
    search = request.args.get('search', '')
    if search:
        filtered_logs = [log for log in filtered_logs if 
                        search.lower() in (log.get('event', '').lower() or '') or
                        search.lower() in (log.get('person', '').lower() or '') or
                        search.lower() in (log.get('status', '').lower() or '') or
                        search.lower() in (log.get('description', '').lower() or '')]
    
    event_type = request.args.get('event_type', '')
    if event_type:
        filtered_logs = [log for log in filtered_logs if log.get('event') == event_type]
    
    start_date = request.args.get('start_date', '')
    if start_date:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        filtered_logs = [log for log in filtered_logs if 
                        datetime.datetime.strptime(log.get('timestamp', '').split()[0], '%Y-%m-%d') >= start_date]
    
    end_date = request.args.get('end_date', '')
    if end_date:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        filtered_logs = [log for log in filtered_logs if 
                        datetime.datetime.strptime(log.get('timestamp', '').split()[0], '%Y-%m-%d') <= end_date]
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    logs = filtered_logs[start_idx:end_idx]
    
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    today_count = sum(1 for log in security_logs if log.get('timestamp', '').startswith(today_date))
    unauthorized_count = sum(1 for log in security_logs if log.get('status') == 'Unauthorized')
    warning_count = sum(1 for log in security_logs if log.get('status') == 'Warning')
    total_count = len(security_logs)
    
    total_logs = len(filtered_logs)
    total_pages = (total_logs + per_page - 1) // per_page
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total_logs,
        'start': start_idx + 1 if logs else 0,
        'end': min(end_idx, total_logs),
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1,
        'next_page': page + 1,
        'total_pages': total_pages
    }
    
    return render_template('logs.html', 
                          logs=logs, 
                          pagination=pagination,
                          today_count=today_count,
                          unauthorized_count=unauthorized_count,
                          warning_count=warning_count,
                          total_count=total_count,
                          filters={'search': search, 'event_type': event_type, 
                                  'start_date': start_date, 'end_date': end_date})

@app.route('/view_log/<log_id>')
def view_log(log_id):
    """View details of a specific log entry."""
    try:
        log = next((log for log in security_logs if log['id'] == log_id), None)
        if log:
            return jsonify(log)
        return jsonify({"error": "Log not found"}), 404
    except Exception as e:
        logging.error(f"Error viewing log {log_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/export_logs')
def export_logs():
    """Export security logs to CSV."""
    try:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Timestamp', 'Event', 'Person', 'Status', 'Confidence', 'Description', 'Metadata'])
        for log in security_logs:
            writer.writerow([
                log.get('timestamp', ''),
                log.get('event', ''),
                log.get('person', 'Unknown'),
                log.get('status', ''),
                log.get('confidence', 0.0),
                log.get('description', ''),
                json.dumps(log.get('metadata', {}))
            ])
        output.seek(0)
        filename = f"security_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logging.error(f"Error exporting logs: {e}")
        flash(f"Error exporting logs: {str(e)}", "error")
        return redirect(url_for('logs_page'))

@app.route('/manage_faces')
def manage_faces():
    """Manage registered faces."""
    try:
        face_db_path = 'static/faces/face_db.json'
        face_db = json.load(open(face_db_path, 'r')) if os.path.exists(face_db_path) else {}
        faces = []
        for face_id, data in face_db.items():
            image_path = f"static/faces/{face_id}.jpg"
            if os.path.exists(image_path):
                faces.append({
                    'id': face_id,
                    'name': data.get('name', 'Unknown'),
                    'access_level': data.get('access_level', 'limited'),
                    'relationship': data.get('relationship', ''),
                    'registered_at': data.get('registered_at', ''),
                    'image_path': image_path
                })
        return render_template('manage_faces.html', faces=faces)
    except Exception as e:
        logging.error(f"Error loading faces: {e}")
        flash(f"Error loading faces: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/delete_face/<face_id>', methods=['POST'])
def delete_face(face_id):
    """Delete a registered face."""
    try:
        image_path = f"static/faces/{face_id}.jpg"
        if os.path.exists(image_path):
            os.remove(image_path)
        
        face_db_path = 'static/faces/face_db.json'
        if os.path.exists(face_db_path):
            with open(face_db_path, 'r') as f:
                face_db = json.load(f)
            name = face_db.get(face_id, {}).get('name', 'Unknown')
            if face_id in face_db:
                del face_db[face_id]
            with open(face_db_path, 'w') as f:
                json.dump(face_db, f)
        
        load_known_faces()
        add_security_log("Face Deleted", name, 1.0, "System")
        flash(f"Face for {name} deleted successfully.", "success")
        return redirect(url_for('manage_faces'))
    except Exception as e:
        logging.error(f"Error deleting face: {e}")
        flash(f"Error deleting face: {str(e)}", "error")
        return redirect(url_for('manage_faces'))

@app.route('/trigger_alarm', methods=['POST'])
def trigger_alarm():
    """Trigger the security alarm."""
    try:
        add_security_log("Alarm Triggered", None, 0.0, "Alert")
        return jsonify({"success": True, "message": "Alarm triggered successfully"})
    except Exception as e:
        logging.error(f"Error triggering alarm: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/learning_stats')
def learning_stats():
    """View learning system statistics."""
    try:
        stats = {
            "continuous_learning": continuous_learning,
            "learning_threshold": learning_threshold,
            "max_samples_per_face": max_samples_per_face,
            "learning_rate": learning_rate,
            "faces": []
        }
        for name in known_face_names:
            face_stats = {"name": name, "samples": 0, "last_updated": "Never"}
            samples_path = f'static/faces/samples/{name.replace(" ", "_")}.h5'
            if os.path.exists(samples_path):
                try:
                    with h5py.File(samples_path, 'r') as f:
                        face_stats["samples"] = len(f['samples'][:])
                except Exception:
                    pass
            if name in last_face_update:
                last_update = datetime.datetime.fromtimestamp(last_face_update[name])
                face_stats["last_updated"] = last_update.strftime("%Y-%m-%d %H:%M:%S")
            stats["faces"].append(face_stats)
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error retrieving learning stats: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/configure_learning', methods=['GET', 'POST'])
def configure_learning():
    """Configure continuous learning parameters."""
    if request.method == 'POST':
        try:
            global learning_threshold, learning_rate, max_samples_per_face, min_learning_interval
            learning_threshold = float(request.form.get('learning_threshold', learning_threshold))
            learning_rate = float(request.form.get('learning_rate', learning_rate))
            max_samples_per_face = int(request.form.get('max_samples_per_face', max_samples_per_face))
            min_learning_interval = int(request.form.get('min_learning_interval', min_learning_interval))
            
            flash("Learning parameters updated successfully!", "success")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Error configuring learning: {e}")
            flash(f"Error configuring learning: {str(e)}", "error")
    
    return render_template('configure_learning.html', 
                          learning_threshold=learning_threshold,
                          learning_rate=learning_rate,
                          max_samples_per_face=max_samples_per_face,
                          min_learning_interval=min_learning_interval)

@app.route('/configure_object_detection', methods=['GET', 'POST'])
def configure_object_detection():
    """Configure object detection parameters."""
    if request.method == 'POST':
        try:
            global object_confidence_threshold, target_object_classes
            object_confidence_threshold = float(request.form.get('confidence_threshold', object_confidence_threshold))
            classes = request.form.get('target_classes', '')
            target_object_classes = [c.strip() for c in classes.split(',')] if classes else []
            flash("Object detection parameters updated successfully!", "success")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Error configuring object detection: {e}")
            flash(f"Error configuring object detection: {str(e)}", "error")
    
    return render_template('configure_object_detection.html', 
                          confidence_threshold=object_confidence_threshold,
                          target_classes=', '.join(target_object_classes))

@app.route('/configure_face_recognition', methods=['GET', 'POST'])
def configure_face_recognition():
    """Configure face recognition parameters."""
    if request.method == 'POST':
        try:
            global face_recognition_model, face_confidence_threshold
            face_recognition_model = request.form.get('model', face_recognition_model)
            face_confidence_threshold = float(request.form.get('confidence_threshold', face_confidence_threshold))
            flash("Face recognition parameters updated successfully!", "success")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Error configuring face recognition: {e}")
            flash(f"Error configuring face recognition: {str(e)}", "error")
    
    return render_template('configure_face_recognition.html', 
                          model=face_recognition_model,
                          confidence_threshold=face_confidence_threshold)

def cleanup():
    """Clean up resources on exit."""
    global camera, yolo_model, mask_net, net
    logging.info("Cleaning up resources...")
    if camera is not None:
        camera.release()
        camera = None
    if yolo_model is not None:
        yolo_model = None
    if mask_net is not None:
        mask_net = None
    if net is not None:
        net = None
    logging.info("Cleanup complete")

def safe_release(obj):
    """Safely release memory for an object."""
    try:
        if obj is not None:
            del obj
    except Exception as e:
        logging.error(f"Error releasing object: {e}")
    return None

# Initialize system
load_known_faces()
load_security_logs()
import atexit
atexit.register(cleanup)
detection_thread = threading.Thread(target=process_detections, daemon=True)
detection_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))
    