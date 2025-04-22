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

# Directories for storing registered and unknown faces
registered_faces_dir = "registered_faces"
unknown_faces_dir = "unknown_faces"
os.makedirs(registered_faces_dir, exist_ok=True)
os.makedirs(unknown_faces_dir, exist_ok=True)

# SQLite database setup
db_file = "detection_records.db"
conn = sqlite3.connect(db_file)
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

# Function to register a new face
def register_face(name):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Registering face for {name}. Look at the camera.")
    face_images = []

    while len(face_images) < 5:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]
            face_images.append(face_image)
            print(f"Captured {len(face_images)} face(s).")

    cap.release()
    cv2.destroyAllWindows()

    if len(face_images) < 5:
        print("Face registration failed. Please try again.")
        return

    for i, face_image in enumerate(face_images):
        filepath = os.path.join(registered_faces_dir, f"{name}_{i}.jpg")
        cv2.imwrite(filepath, face_image)
        print(f"Saved: {filepath}")

# Function to log events into the database
def log_event(detected_object, confidence, face_name=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO detections (timestamp, detected_object, confidence, face_name)
    VALUES (?, ?, ?, ?)
    ''', (timestamp, detected_object, confidence, face_name))
    conn.commit()
    print(f"Logged: {detected_object} (Confidence: {confidence:.2f}, Face: {face_name})")

# Function to save unknown faces
def save_unknown_face(face_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
    cv2.imwrite(filepath, face_image)
    print(f"Unknown face saved: {filepath}")

def detect_mask(frame, conf_thresh=0.5):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(260, 260))
    mask_net.setInput(blob)

    # Perform forward pass and get the output
    outputs = mask_net.forward()

    # Debugging output
    print(f"outputs shape: {outputs.shape}")
    print(outputs)

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


def detection_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Perform weapon detection with YOLO
            results = weapon_model(frame)
            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box
                    label = result.names[int(cls)]
                    confidence = float(conf)

                    # Draw bounding box for detected objects (e.g., weapons)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Log detected object in the database
                    log_event(label, confidence, face_name="")

            # Face recognition and mask detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index]

                # Detect mask within the face region
                face_region = frame[top:bottom, left:right]
                face_region, mask_detected = detect_mask(face_region)

                # Save unknown faces
                if name == "Unknown":
                    save_unknown_face(face_region)

                # Determine annotation color based on mask status
                if mask_detected:
                    color = (0, 0, 255)  # Green for mask
                    mask_status = "Mask"
                else:
                  in `start_detection` to handle SSD model output correctly. The `detections` array is processed to extract `confidence` and `class_id` for each detection, using `id2class` for labeling (“Mask” or “NoMask”).
                                                                                                                                                                                                                                ^
SyntaxError: invalid character '“' (U+201C)  color = (0, 255, 0)  # Red for no mask
                    mask_status = "No Mask"

                # Draw bounding box and labels for faces
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name}: {mask_status}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Log the face detection event in the database
                log_event("person", 1.0, face_name=f"{name} ({mask_status})")

            # Display the resulting frame with all annotations
            cv2.imshow('Home Security System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()


# Entry point
# Entry point
if __name__ == "__main__":
    print("1. Register Face")
    print("2. Start Detection Loop")
    choice = input("Enter choice: ")

    if choice == "1":
        name = input("Enter your name: ")
        register_face(name)
    elif choice == "2":
        detection_loop()
    else:
        print("Invalid choice. Please enter 1 or 2.")