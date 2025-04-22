import cv2
import os
import sqlite3
import numpy as np
import face_recognition
from ultralytics import YOLO
from datetime import datetime
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import re
import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)  # or GPIO.BOARD depending on your pin reference
GPIO.setwarnings(False)

# Set up GPIO 16 as output
buzzer_pin = 16
GPIO.setup(buzzer_pin, GPIO.OUT)
def buzzer_on():
    GPIO.output(buzzer_pin, GPIO.HIGH)  # Turn on buzzer

def buzzer_off():
    GPIO.output(buzzer_pin, GPIO.LOW)   # Turn off buzzer

# Assuming utils are available; replace with actual implementations if needed
def generate_anchors(feature_map_sizes, anchor_scales, anchor_ratios):
    anchors = []
    for idx, fmap_size in enumerate(feature_map_sizes):
        fmap_w, fmap_h = fmap_size
        scale = anchor_scales[idx]
        ratios = anchor_ratios[idx]
        for i in range(fmap_w):
            for j in range(fmap_h):
                cx = (j + 0.5) / fmap_w
                cy = (i + 0.5) / fmap_h
                for ratio in ratios:
                    for s in scale:
                        w = s * np.sqrt(ratio)
                        h = s / np.sqrt(ratio)
                        anchors.append([cx, cy, w, h])
    return np.array(anchors)

# Placeholder for anchor_decode and nms (implement or source from model repo)
def decode_bbox(*args, **kwargs):
    return np.zeros((0, 4))  # Replace with actual implementation
def single_class_non_max_suppression(*args, **kwargs):
    return []  # Replace with actual implementation

# CustomTkinter setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Directories
REGISTERED_DIR = "registered_faces"
UNKNOWN_DIR = "unknown_faces"
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# SQLite setup
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

# Load models
try:
    weapon_model = YOLO("yolov5nu.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    weapon_model = None

try:
    mask_net = cv2.dnn.readNet('models/face_mask_detection.caffemodel', 'models/face_mask_detection.prototxt')
except Exception as e:
    print(f"Error loading mask model: {e}")
    mask_net = None

# Generate anchors
anchors = generate_anchors(
    [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]],
    [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]],
    [[1, 0.62, 0.42]] * 5
)
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0, 0))

# Face recognition setup
known_face_encodings, known_face_names = [], []

def load_registered_faces():
    encodings, names = [], []
    for filename in os.listdir(REGISTERED_DIR):
        if filename.endswith((".jpg", ".png")):
            filepath = os.path.join(REGISTERED_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                encodings.append(face_recognition.face_encodings(image, face_locations)[0])
                names.append(os.path.splitext(filename)[0].rsplit("_", 1)[0])
    return encodings, names

known_face_encodings, known_face_names = load_registered_faces()

# Database logging
def log_event(detected_object, confidence, face_name=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO detections (timestamp, detected_object, confidence, face_name) VALUES (?, ?, ?, ?)',
                   (timestamp, detected_object, confidence, face_name))
    conn.commit()

def save_unknown_face(face_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(UNKNOWN_DIR, f"unknown_{timestamp}.jpg")
    cv2.imwrite(filepath, face_image)
    buzzer_on()
    time.sleep(3)  # Buzzer on for 1 second
    buzzer_off()
    buzzer_on()
    time.sleep(3)  # Buzzer on for 1 second
    buzzer_off()
    # Optional: Clean up GPIO
    GPIO.cleanup()

# Mask detection
def detect_mask(frame, conf_thresh=0.5):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(260, 260))
    mask_net.setInput(blob)
    outputs = mask_net.forward()
    mask_detected = False
    if len(outputs.shape) != 4 or outputs.shape[2] == 0:
        return frame, mask_detected
    for i in range(outputs.shape[2]):
        detection = outputs[0, 0, i]
        confidence = float(detection[2])
        if confidence > conf_thresh:
            xmin = int(detection[3] * width)
            ymin = int(detection[4] * height)
            xmax = int(detection[5] * width)
            ymax = int(detection[6] * height)
            class_id = int(detection[1])
            if class_id not in id2class:
                continue
            label = id2class[class_id]
            color = colors[class_id]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if label == "Mask":
                mask_detected = True
    return frame, mask_detected

# Detection loop with GUI integration
class DetectionThread(threading.Thread):
    def __init__(self, video_label, status_label, alert_label, main_frame):
        super().__init__()
        self.running = True
        self.video_label = video_label
        self.status_label = status_label
        self.alert_label = alert_label
        self.main_frame = main_frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_label.configure(text="Error: Webcam unavailable")
            messagebox.showerror("Error", "Cannot access webcam.")
            return

        self.status_label.configure(text="Combined Detection... Press Stop to exit")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_label.configure(text="Error: Frame capture failed")
                break

            # Weapon detection
            if weapon_model:
                results = weapon_model(frame)
                for result in results:
                    for box in result.boxes.data:
                        x1, y1, x2, y2, conf, cls = box
                        label = result.names[int(cls)]
                        confidence = float(conf)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        log_event(label, confidence, "")
                        self.alert_label.configure(text=f"ALERT: {label} detected!", text_color="red")

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

                face_region = frame[top:bottom, left:right]
                face_region, mask_detected = detect_mask(face_region)
                if name == "Unknown":
                    save_unknown_face(face_region)

                color = (0, 255, 0) if mask_detected else (0, 0, 255)
                mask_status = "Mask" if mask_detected else "No Mask"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name}: {mask_status}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                log_event("person", 1.0, f"{name} ({mask_status})")

            # Display in GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

        cap.release()
        self.status_label.configure(text="Detection stopped")
        self.alert_label.configure(text="")

    def stop(self):
        self.running = False

# GUI functions
def register_face(status_label, main_frame):
    def capture():
        name = entry.get().strip()
        name = re.sub(r'[^\w\-]', '', name)
        if not name:
            messagebox.showerror("Error", "Name is required.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam.")
            return

        status_label.configure(text=f"Registering {name}... Press Q to quit")
        face_images = []
        while len(face_images) < 5:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = frame[top:bottom, left:right]
                face_images.append(face_img)
            cv2.imshow("Registering...", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

        for i, face_img in enumerate(face_images):
            filepath = os.path.join(REGISTERED_DIR, f"{name}_{i}.jpg")
            cv2.imwrite(filepath, face_img)
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)

        status_label.configure(text=f"Registered {name}!")
        messagebox.showinfo("Success", f"{name} registered!")
        top.destroy()

    clear_main_frame(main_frame)
    top = ctk.CTkToplevel()
    top.title("Register Face")
    top.geometry("300x200")
    ctk.CTkLabel(top, text="Enter Name:", font=("Roboto", 14)).pack(pady=10)
    entry = ctk.CTkEntry(top, width=200, font=("Roboto", 12))
    entry.pack(pady=10)
    ctk.CTkButton(top, text="Capture Face", command=capture, font=("Roboto", 12)).pack(pady=10)

def start_detection(status_label, alert_label, main_frame):
    global detection_thread
    clear_main_frame(main_frame)
    video_label = ctk.CTkLabel(main_frame, text="")
    video_label.pack(pady=10)
    stop_button = ctk.CTkButton(main_frame, text="Stop Detection", command=lambda: stop_detection(), font=("Roboto", 12))
    stop_button.pack(pady=10)
    detection_thread = DetectionThread(video_label, status_label, alert_label, main_frame)
    detection_thread.start()

def stop_detection():
    global detection_thread
    if detection_thread:
        detection_thread.stop()
        detection_thread = None

def show_faces(main_frame):
    global known_face_encodings, known_face_names
    clear_main_frame(main_frame)
    canvas = ctk.CTkCanvas(main_frame)
    scroll = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview)
    frame = ctk.CTkFrame(canvas)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    row, col = 0, 0
    for fname in os.listdir(REGISTERED_DIR):
        if fname.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(REGISTERED_DIR, fname))
            img.thumbnail((100, 100))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))
            name = fname.split("_")[0]
            face_frame = ctk.CTkFrame(frame)
            label = ctk.CTkLabel(face_frame, image=imgtk, text="")
            label.pack(pady=5)
            ctk.CTkLabel(face_frame, text=name, font=("Roboto", 12)).pack()
            ctk.CTkButton(face_frame, text="Delete", command=lambda f=fname: delete_face(f, main_frame), font=("Roboto", 12)).pack(pady=5)
            face_frame.grid(row=row, column=col, padx=10, pady=10)
            col += 1
            if col > 3:
                col = 0
                row += 1

def delete_face(fname, main_frame):
    os.remove(os.path.join(REGISTERED_DIR, fname))
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_registered_faces()
    messagebox.showinfo("Success", f"{fname} deleted!")
    show_faces(main_frame)

def review_unknown_faces(status_label, main_frame):
    clear_main_frame(main_frame)
    canvas = ctk.CTkCanvas(main_frame)
    scroll = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview)
    frame = ctk.CTkFrame(canvas)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    row, col = 0, 0
    for fname in os.listdir(UNKNOWN_DIR):
        if fname.endswith(('.jpg', '.png')):
            img = Image.open(os.path.join(UNKNOWN_DIR, fname))
            img.thumbnail((100, 100))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(100, 100))
            face_frame = ctk.CTkFrame(frame)
            label = ctk.CTkLabel(face_frame, image=imgtk, text="")
            label.pack(pady=5)
            entry = ctk.CTkEntry(face_frame, placeholder_text="Enter name", font=("Roboto", 12))
            entry.pack(pady=5)
            ctk.CTkButton(face_frame, text="Register",
                          command=lambda f=fname, e=entry: register_unknown(f, e, main_frame, status_label),
                          font=("Roboto", 12)).pack(pady=5)
            face_frame.grid(row=row, column=col, padx=10, pady=10)
            col += 1
            if col > 3:
                col = 0
                row += 1

def register_unknown(fname, entry, main_frame, status_label):
    name = entry.get().strip()
    name = re.sub(r'[^\w\-]', '', name)
    if not name:
        messagebox.showerror("Error", "Name is required.")
        return
    src_path = os.path.join(UNKNOWN_DIR, fname)
    dst_path = os.path.join(REGISTERED_DIR, f"{name}_0.jpg")
    os.rename(src_path, dst_path)
    image = face_recognition.load_image_file(dst_path)
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    status_label.configure(text=f"Registered {name} from unknown")
    messagebox.showinfo("Success", f"{name} registered!")
    review_unknown_faces(status_label, main_frame)

def show_reports(main_frame):
    clear_main_frame(main_frame)
    tree = tk.ttk.Treeview(main_frame, columns=("Time", "Object", "Confidence", "Name"), show="headings")
    for col in ("Time", "Object", "Confidence", "Name"):
        tree.heading(col, text=col)
        tree.column(col, width=150)
    cursor.execute("SELECT timestamp, detected_object, confidence, face_name FROM detections ORDER BY timestamp DESC LIMIT 100")
    for row in cursor.fetchall():
        tree.insert("", "end", values=row)
    tree.pack(fill="both", expand=True)

def show_settings(status_label, main_frame):
    clear_main_frame(main_frame)
    ctk.CTkLabel(main_frame, text="Settings", font=("Roboto", 16, "bold")).pack(pady=10)
    ctk.CTkLabel(main_frame, text="Theme:", font=("Roboto", 12)).pack(pady=5)
    theme_option = ctk.CTkOptionMenu(main_frame, values=["Dark", "Light", "System"], command=lambda mode: ctk.set_appearance_mode(mode))
    theme_option.set("Dark")
    theme_option.pack(pady=5)

def clear_main_frame(main_frame):
    for widget in main_frame.winfo_children():
        widget.destroy()

def cleanup(root):
    global detection_thread
    if detection_thread:
        detection_thread.stop()
    conn.close()
    cv2.destroyAllWindows()
    root.quit()

# Main GUI
def main_ui():
    global detection_thread
    detection_thread = None
    root = ctk.CTk()
    root.title("Home Security System")
    root.geometry("900x600")

    # Sidebar
    sidebar = ctk.CTkFrame(root, width=200, corner_radius=0)
    sidebar.pack(side="left", fill="y")
    ctk.CTkLabel(sidebar, text="Home Security", font=("Roboto", 16, "bold")).pack(pady=20)

    buttons = [
        ("Register member", lambda: register_face(status_label, main_frame), "Register a new member"),
        ("Start monitoring", lambda: start_detection(status_label, alert_label, main_frame), "Start combined detection"),
        ("Family member", lambda: show_faces(main_frame), "View registered faces"),
        ("Review Unknown", lambda: review_unknown_faces(status_label, main_frame), "Review unknown faces"),
        ("Reports", lambda: show_reports(main_frame), "View detection reports"),
        ("Settings", lambda: show_settings(status_label, main_frame), "Adjust settings")
    ]

    for text, command, tooltip in buttons:
        btn = ctk.CTkButton(sidebar, text=text, command=command, font=("Roboto", 12), height=40)
        btn.pack(pady=5, padx=10, fill="x")
        btn.bind("<Enter>", lambda e, t=tooltip: status_label.configure(text=t))
        btn.bind("<Leave>", lambda e: status_label.configure(text="Idle"))

    # Main Frame
    main_frame = ctk.CTkFrame(root, corner_radius=10)
    main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    ctk.CTkLabel(main_frame, text="Welcome to Home Security", font=("Roboto", 20, "bold")).pack(pady=20)
    ctk.CTkLabel(main_frame, text="Select an option from the sidebar.", font=("Roboto", 14)).pack(pady=10)

    # Status Bar
    status_frame = ctk.CTkFrame(root, height=30, corner_radius=0)
    status_frame.pack(side="bottom", fill="x")
    status_label = ctk.CTkLabel(status_frame, text="Idle", font=("Roboto", 12))
    status_label.pack(side="left", padx=10)
    alert_label = ctk.CTkLabel(status_frame, text="", font=("Roboto", 12), text_color="red")
    alert_label.pack(side="right", padx=10)

    root.protocol("WM_DELETE_WINDOW", lambda: cleanup(root))
    root.mainloop()

if __name__ == "__main__":
    main_ui()