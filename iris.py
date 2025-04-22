from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
import cv2
import os
import sqlite3
import face_recognition
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Directories
REGISTERED_FACES_DIR = "registered_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Database setup
DB_FILE = "detection_records.db"
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

# Models
weapon_model = YOLO('yolov5nu.pt')
camera = cv2.VideoCapture(1)

# Known Faces
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []

# State Control
is_running = False


def load_registered_faces():
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
    encodings = []
    names = []
    for filename in os.listdir(REGISTERED_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(REGISTERED_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    names.append(os.path.splitext(filename)[0].rsplit("_", 1)[0])
    KNOWN_FACE_ENCODINGS = encodings
    KNOWN_FACE_NAMES = names


def generate_frames():
    global is_running
    while is_running:
        success, frame = camera.read()
        if not success:
            break

        # Weapon Detection
        results = weapon_model(frame)
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                label = result.names[int(cls)]
                confidence = float(conf)
                if label in ["gun", "knife", "pistol"]:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cursor.execute('''
                        INSERT INTO detections (timestamp, detected_object, confidence)
                        VALUES (?, ?, ?)
                    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence))
                    conn.commit()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    plot_url = generate_plot()
    return render_template('index.html', plot_url=plot_url, is_running=is_running)


@app.route('/video_feed')
def video_feed():
    global is_running
    is_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_detection')
def stop_detection():
    global is_running
    is_running = False
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Error: Could not access camera."

        count = 0
        while count < 5:
            ret, frame = cap.read()
            if ret:
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    count += 1
                    filename = os.path.join(REGISTERED_FACES_DIR, f"{name}_{count}.jpg")
                    cv2.imwrite(filename, frame)

        cap.release()
        load_registered_faces()
        return redirect(url_for('index'))
    return render_template('detection.html')


@app.route('/detections')
def detections():
    cursor.execute("SELECT * FROM detections")
    rows = cursor.fetchall()
    return jsonify(rows)


@app.route('/registered_faces/<filename>')
def registered_faces(filename):
    return send_from_directory(REGISTERED_FACES_DIR, filename)


@app.route('/unknown_faces/<filename>')
def unknown_faces(filename):
    return send_from_directory(UNKNOWN_FACES_DIR, filename)


def generate_plot():
    cursor.execute('SELECT detected_object FROM detections')
    data = cursor.fetchall()
    objects = [row[0] for row in data]
    object_counts = {obj: objects.count(obj) for obj in set(objects)}

    plt.figure(figsize=(8, 4))
    plt.bar(object_counts.keys(), object_counts.values(), color='skyblue')
    plt.xlabel('Detected Objects')
    plt.ylabel('Count')
    plt.title('Detection Log Summary')
    plt.xticks(rotation=45)

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()


if __name__ == '__main__':
    load_registered_faces()
    app.run(host='0.0.0.0', port=5100, debug=True)
