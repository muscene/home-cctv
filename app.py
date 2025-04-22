import cv2
import os
import sqlite3
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from datetime import datetime
from ultralytics import YOLO
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
import threading
import time
import json

app = Flask(__name__)

# Directories for storing registered and unknown faces
registered_faces_dir = "static/registered_faces"
unknown_faces_dir = "static/unknown_faces"
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
try:
    weapon_model = YOLO('yolov5nu.pt')  # Ensure this model file exists
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    weapon_model = None

# Load pre-trained face mask detection model
try:
    mask_net = cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')
except Exception as e:
    print(f"Error loading mask detection model: {e}")
    mask_net = None

# Generate anchors for face mask detection
anchors = generate_anchors(
    [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]],
    [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]],
    [[1, 0.62, 0.42]] * 5
)
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0, 0))

# Global variables
camera = None
camera_status = "Disconnected"
frame_buffer = None
detection_thread = None
detection_running = False
recent_events = []
known_face_encodings = []
known_face_names = []

# Function to load registered faces
def load_registered_faces():
    global known_face_encodings, known_face_names
    encodings = []
    names = []
    for filename in os.listdir(registered_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(registered_faces_dir, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    if face_encodings:
                        encodings.append(face_encodings[0])
                        names.append(os.path.splitext(filename)[0].rsplit("_", 1)[0])
            except Exception as e:
                print(f"Error loading face {filename}: {e}")
    
    known_face_encodings = encodings
    known_face_names = names
    print(f"Loaded {len(encodings)} registered faces")
    return encodings, names

# Initialize camera
def init_camera():
    global camera, camera_status
    try:
        camera = cv2.VideoCapture(0)  # Try default camera
        if not camera.isOpened():
            camera = cv2.VideoCapture(1)  # Try alternative camera
        
        if camera.isOpened():
            camera_status = "Connected"
            return True
        else:
            camera_status = "Failed to connect"
            return False
    except Exception as e:
        camera_status = f"Error: {str(e)}"
        return False

# Function to log events into the database
def log_event(detected_object, confidence, face_name=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute('''
        INSERT INTO detections (timestamp, detected_object, confidence, face_name)
        VALUES (?, ?, ?, ?)
        ''', (timestamp, detected_object, confidence, face_name))
        conn.commit()
        
        # Add to recent events for real-time updates
        event_message = f"{detected_object} detected"
        if face_name:
            event_message += f" ({face_name})"
        
        recent_events.append({
            "message": event_message,
            "timestamp": timestamp
        })
        
        # Keep only the last 10 events
        if len(recent_events) > 10:
            recent_events.pop(0)
            
        print(f"Logged: {detected_object} (Confidence: {confidence:.2f}, Face: {face_name})")
    except Exception as e:
        print(f"Error logging event: {e}")

# Function to save unknown faces
def save_unknown_face(face_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
    cv2.imwrite(filepath, face_image)
    print(f"Unknown face saved: {filepath}")
    return filepath

def detect_mask(frame, conf_thresh=0.5):
    if mask_net is None:
        return frame, False
        
    try:
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(260, 260))
        mask_net.setInput(blob)

        # Perform forward pass and get the output
        outputs = mask_net.forward()
        mask_detected = False

        # Ensure outputs have the correct shape
        if len(outputs.shape) != 4 or outputs.shape[2] == 0:
            return frame, mask_detected

        # Process each detection in the outputs
        for i in range(outputs.shape[2]):
            detection = outputs[0, 0, i]
            confidence = float(detection[2])
            if confidence > conf_thresh:
                xmin = int(detection[3] * width)
                ymin = int(detection[4] * height)
                xmax = int(detection[5] * width)
                ymax = int(detection[6] * height)
                class_id = int(detection[1])

                # Ensure class_id is valid
                if class_id not in id2class:
                    continue

                label = id2class[class_id]
                color = colors[class_id]

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if label == "Mask":
                    mask_detected = True

        return frame, mask_detected
    except Exception as e:
        print(f"Error in mask detection: {e}")
        return frame, False

def detection_process():
    global camera, frame_buffer, detection_running, camera_status
    
    if not camera or not camera.isOpened():
        if not init_camera():
            print("Failed to initialize camera for detection")
            detection_running = False
            return
    
    try:
        while detection_running:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame.")
                camera_status = "Feed interrupted"
                time.sleep(1)
                continue

            processed_frame = frame.copy()
            
            # Perform weapon detection with YOLO
            if weapon_model:
                try:
                    results = weapon_model(frame)
                    for result in results:
                        for box in result.boxes.data:
                            x1, y1, x2, y2, conf, cls = box
                            label = result.names[int(cls)]
                            confidence = float(conf)

                            # Draw bounding box for detected objects (e.g., weapons)
                            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(processed_frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            # Log detected object in the database
                            log_event(label, confidence)
                except Exception as e:
                    print(f"Error in weapon detection: {e}")

            # Face recognition and mask detection
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding) if known_face_encodings else []
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding) if known_face_encodings else []
                    name = "Unknown"

                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        name = known_face_names[best_match_index]

                    # Detect mask within the face region
                    if bottom - top > 0 and right - left > 0:
                        face_region = processed_frame[top:bottom, left:right]
                        face_region, mask_detected = detect_mask(face_region)
                        processed_frame[top:bottom, left:right] = face_region
                    else:
                        mask_detected = False

                    # Save unknown faces
                    if name == "Unknown":
                        if bottom - top > 0 and right - left > 0:
                            face_img = frame[top:bottom, left:right]
                            save_unknown_face(face_img)

                    # Determine annotation color based on mask status
                    if mask_detected:
                        color = (0, 255, 0)  # Green for mask
                        mask_status = "Mask"
                    else:
                        color = (0, 0, 255)  # Red for no mask
                        mask_status = "No Mask"

                    # Draw bounding box and labels for faces
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
                    label = f"{name}: {mask_status}"
                    cv2.putText(processed_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Log the face detection event in the database
                    log_event("person", 1.0, face_name=f"{name} ({mask_status})")
            except Exception as e:
                print(f"Error in face recognition: {e}")

            # Update the frame buffer for streaming
            frame_buffer = processed_frame
            time.sleep(0.03)  # ~30 FPS

    except Exception as e:
        print(f"Error in detection process: {e}")
    finally:
        detection_running = False
        if camera and camera.isOpened():
            camera.release()
        camera_status = "Disconnected"

# Generate camera frames for streaming
def generate_frames():
    global frame_buffer, camera_status
    while True:
        if frame_buffer is not None:
            ret, buffer = cv2.imencode('.jpg', frame_buffer)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Return a placeholder when no frame is available
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, camera_status, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # ~30 FPS

# Generate SSE events for real-time updates
def generate_status_events():
    global camera_status, recent_events
    last_event_index = 0
    
    while True:
        # Send current status
        data = json.dumps({"status": camera_status})
        yield f"event: status\ndata: {data}\n\n"
        
        # Send any new events
        if recent_events and last_event_index < len(recent_events):
            event_data = json.dumps(recent_events[last_event_index])
            yield f"event: event\ndata: {event_data}\n\n"
            last_event_index = len(recent_events)
            
        time.sleep(1)

# Routes
@app.route('/')
def index():
    global camera_status
    return render_template('index.html', camera_status=camera_status)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_stream')
def status_stream():
    return Response(generate_status_events(),
                    mimetype='text/event-stream')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera, camera_status, detection_running, detection_thread
    data = request.json
    action = data.get('action', '')
    
    if action == 'connect':
        if not detection_running:
            detection_running = True
            detection_thread = threading.Thread(target=detection_process)
            detection_thread.daemon = True
            detection_thread.start()
            return jsonify({"success": True, "status": "Connecting..."})
        else:
            return jsonify({"success": False, "status": camera_status})
    elif action == 'disconnect':
        detection_running = False
        if detection_thread:
            detection_thread.join(timeout=1.0)
        if camera and camera.isOpened():
            camera.release()
        camera_status = "Disconnected"
        return jsonify({"success": True, "status": camera_status})
    
    return jsonify({"success": False, "status": camera_status})

@app.route('/trigger_alarm', methods=['POST'])
def trigger_alarm():
    # In a real app, this would trigger an actual alarm
    log_event("Alarm", 1.0, face_name="Manually Triggered")
    return jsonify({"success": True, "message": "Alarm triggered"})

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global camera, frame_buffer
    
    if frame_buffer is not None:
        # Save the current frame
        os.makedirs('static/captures', exist_ok=True)
        filename = f"capture_{int(time.time())}.jpg"
        filepath = os.path.join('static/captures', filename)
        cv2.imwrite(filepath, frame_buffer)
        return jsonify({"success": True, "image_url": url_for('static', filename=f'captures/{filename}')})
    
    return jsonify({"success": False, "message": "No frame available"})

@app.route('/register_face', methods=['POST'])
def register_face():
    name = request.form.get('name')
    relationship = request.form.get('relationship')
    access_level = request.form.get('access_level')
    face_image_url = request.form.get('face_image')
    
    if not name or not face_image_url:
        return redirect(url_for('register', error="Missing required information"))
    
    try:
        # Extract the filename from the URL
        image_path = face_image_url.split('?')[0]  # Remove any query parameters
        if image_path.startswith('/'):
            image_path = image_path[1:]  # Remove leading slash
            
        # Read the captured image
        image = cv2.imread(image_path)
        if image is None:
            return redirect(url_for('register', error="Failed to read image"))
            
        # Save the face image with the person's name
        filename = f"{name}_{int(time.time())}.jpg"
        filepath = os.path.join(registered_faces_dir, filename)
        cv2.imwrite(filepath, image)
        
        # Reload face encodings
        load_registered_faces()
        
        # Log the registration
        log_event("Registration", 1.0, face_name=name)
        
        return redirect(url_for('index', success=f"Successfully registered {name}"))
    except Exception as e:
        print(f"Error registering face: {e}")
        return redirect(url_for('register', error=f"Error: {str(e)}"))

@app.route('/logs')
def logs_page():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Get logs from database
    try:
        cursor.execute('''
        SELECT id, timestamp, detected_object, confidence, face_name
        FROM detections
        ORDER BY timestamp DESC
        ''')
        all_logs = cursor.fetchall()
        
        # Convert to list of dictionaries
        logs = []
        for log in all_logs:
            logs.append({
                "id": log[0],
                "timestamp": log[1],
                "event": log[2],
                "confidence": log[3],
                "person": log[4] if log[4] else None,
                "status": "Authorized" if "Unknown" not in (log[4] or "") else "Unauthorized"
            })
        
        # Filter logs based on query parameters
        filtered_logs = logs.copy()
        search = request.args.get('search', '')
        if search:
            filtered_logs = [log for log in filtered_logs if search.lower() in str(log).lower()]
        
        # Pagination
        start = (page - 1) * per_page
        end = min(start + per_page, len(filtered_logs))
        paginated_logs = filtered_logs[start:end]
        
        pagination = {
            'page': page,
            'per_page': per_page,
            'total': len(filtered_logs),
            'start': start + 1 if filtered_logs else 0,
            'end': end,
            'has_prev': page > 1,
            'has_next': end < len(filtered_logs),
            'prev_page': page - 1,
            'next_page': page + 1
        }
        
        return render_template('logs.html', logs=paginated_logs, pagination=pagination)
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return render_template('logs.html', logs=[], pagination={
            'page': 1, 'per_page': 10, 'total': 0, 'start': 0, 'end': 0,
            'has_prev': False, 'has_next': False, 'prev_page': 1, 'next_page': 1
        }, error=str(e))

@app.route('/view_log/<int:log_id>')
def view_log(log_id):
    try:
        cursor.execute('''
        SELECT id, timestamp, detected_object, confidence, face_name
        FROM detections
        WHERE id = ?
        ''', (log_id,))
        log = cursor.fetchone()
        
        if log:
            return jsonify({
                "id": log[0],
                "timestamp": log[1],
                "detected_object": log[2],
                "confidence": log[3],
                "face_name": log[4]
            })
        return "Log not found", 404
    except Exception as e:
        print(f"Error viewing log: {e}")
        return f"Error: {str(e)}", 500

@app.route('/export_logs')
def export_logs():
    try:
        cursor.execute('''
        SELECT id, timestamp, detected_object, confidence, face_name
        FROM detections
        ORDER BY timestamp DESC
        ''')
        logs = cursor.fetchall()
        
        # Convert to list of dictionaries
        result = []
        for log in logs:
            result.append({
                "id": log[0],
                "timestamp": log[1],
                "detected_object": log[2],
                "confidence": log[3],
                "face_name": log[4]
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error exporting logs: {e}")
        return jsonify({"error": str(e)})

# Start the detection thread when the app starts
@app.before_first_request
def before_first_request():
    global detection_running, detection_thread
    
    # Load registered faces
    load_registered_faces()
    
    # Start detection thread
    detection_running = True
    detection_thread = threading.Thread(target=detection_process)
    detection_thread.daemon = True
    detection_thread.start()

if __name__ == '__main__':
    # Load registered faces
    load_registered_faces()
    
    # Start the Flask app
    app.run(debug=True, threaded=True)