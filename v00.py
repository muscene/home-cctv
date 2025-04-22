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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Set CustomTkinter appearance for hackers template
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")  # Base theme, customized below

# Set GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Set up GPIO 16 as output
buzzer_pin = 16
GPIO.setup(buzzer_pin, GPIO.OUT)

def buzzer_on():
    GPIO.output(buzzer_pin, GPIO.HIGH)

def buzzer_off():
    GPIO.output(buzzer_pin, GPIO.LOW)

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'nsanzestevo1@gmail.com',
    'sender_password': 'pribogffrytdmtya',  # App Password, no spaces
    'receiver_email': 'gasasiras103@gmail.com'
}

# Directories
REGISTERED_DIR = "registered_faces"
UNKNOWN_DIR = "unknown_faces"
VIDEO_DIR = "incident_videos"
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

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

# Email sending function
def send_email(subject, body, video_path=None):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['sender_email']
    msg['To'] = EMAIL_CONFIG['receiver_email']
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if video_path and os.path.exists(video_path):
        with open(video_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename={os.path.basename(video_path)}'
        )
        msg.attach(part)

    try:
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.sendmail(
            EMAIL_CONFIG['sender_email'],
            EMAIL_CONFIG['receiver_email'],
            msg.as_string()
        )
        server.quit()
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Generate anchors
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

# Placeholder functions
def decode_bbox(*args, **kwargs):
    return np.zeros((0, 4))

def single_class_non_max_suppression(*args, **kwargs):
    return []

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
    time.sleep(3)
    buzzer_off()
    buzzer_on()
    time.sleep(3)
    buzzer_off()

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

# Detection thread with video recording and email
class DetectionThread(threading.Thread):
    def __init__(self, video_label, status_label, alert_label, main_frame):
        super().__init__()
        self.running = True
        self.video_label = video_label
        self.status_label = status_label
        self.alert_label = alert_label
        self.main_frame = main_frame
        self.last_notification = {}  # Track last notification time for each event type
        self.notification_cooldown = 10  # Seconds between notifications
        self.recording = False
        self.video_writer = None
        self.video_frames = []
        self.incident_start_time = None
        self.incident_type = None

    def start_recording(self, frame, incident_type):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(VIDEO_DIR, f"{incident_type}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (frame.shape[1], frame.shape[0]))
            self.recording = True
            self.video_frames = []
            self.incident_start_time = time.time()
            self.incident_type = incident_type
            self.video_path = video_path

    def stop_recording(self):
        if self.recording:
            for frame in self.video_frames:
                self.video_writer.write(frame)
            self.video_writer.release()
            self.recording = False
            self.video_writer = None
            self.video_frames = []
            # Send email with video
            subject = f"Security Alert: {self.incident_type} Detected"
            body = f"{self.incident_type} was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            threading.Thread(target=send_email, args=(subject, body, self.video_path)).start()
            self.incident_type = None
            self.video_path = None

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_label.configure(text="Error: Webcam unavailable")
            messagebox.showerror("Error", "Cannot access webcam.")
            return

        self.status_label.configure(text="> Scanning... [Press Stop to Terminate]")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_label.configure(text="> Error: Frame Capture Failed")
                break

            current_time = time.time()

            # Weapon detection
            weapon_detected = False
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
                        self.alert_label.configure(text=f"ALERT: {label} DETECTED!", text_color="#FF0000")
                        weapon_detected = True
                        if 'weapon' not in self.last_notification or (current_time - self.last_notification.get('weapon', 0)) > self.notification_cooldown:
                            self.start_recording(frame, 'weapon')
                            self.last_notification['weapon'] = current_time

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
                    if 'unknown_face' not in self.last_notification or (current_time - self.last_notification.get('unknown_face', 0)) > self.notification_cooldown:
                        self.start_recording(frame, 'unknown_face')
                        self.last_notification['unknown_face'] = current_time

                if mask_detected:
                    if 'mask' not in self.last_notification or (current_time - self.last_notification.get('mask', 0)) > self.notification_cooldown:
                        self.start_recording(frame, 'mask')
                        self.last_notification['mask'] = current_time

                color = (0, 255, 0) if mask_detected else (0, 0, 255)
                mask_status = "Mask" if mask_detected else "No Mask"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name}: {mask_status}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                log_event("person", 1.0, f"{name} ({mask_status})")

            # Handle video recording
            if self.recording:
                self.video_frames.append(frame.copy())
                if (current_time - self.incident_start_time) >= 20:  # Record for 20 seconds
                    self.stop_recording()

            # Display in GUI with hackers overlay
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(frame, "> SCANNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            img = Image.fromarray(frame)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

        # Cleanup
        if self.recording:
            self.stop_recording()
        cap.release()
        self.status_label.configure(text="> Detection Terminated")
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

        status_label.configure(text=f"> Registering {name}... [Press Q to Quit]")
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

        status_label.configure(text=f"> Registered {name}!")
        messagebox.showinfo("Success", f"{name} registered!")
        top.destroy()

    clear_main_frame(main_frame)
    top = ctk.CTkToplevel()
    top.title("Register Face")
    top.geometry("300x200")
    top.configure(fg_color="#000000")
    ctk.CTkLabel(top, text="> Enter Name:", font=("Courier New", 14), text_color="#00FF00").pack(pady=10)
    entry = ctk.CTkEntry(top, width=200, font=("Courier New", 12), text_color="#00FF00", fg_color="#1C2526")
    entry.pack(pady=10)
    ctk.CTkButton(top, text="Capture Face", command=capture, font=("Courier New", 12), fg_color="#00FFFF", hover_color="#00FF00", text_color="#000000").pack(pady=10)

def start_detection(status_label, alert_label, main_frame):
    global detection_thread
    clear_main_frame(main_frame)
    video_label = ctk.CTkLabel(main_frame, text="", corner_radius=5, fg_color="#000000")
    video_label.pack(pady=10)
    stop_button = ctk.CTkButton(main_frame, text="> Terminate Detection", command=lambda: stop_detection(), font=("Courier New", 12), fg_color="#00FFFF", hover_color="#00FF00", text_color="#000000")
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
    canvas = ctk.CTkCanvas(main_frame, bg="#000000", highlightthickness=0)
    scroll = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview, fg_color="#1C2526")
    frame = ctk.CTkFrame(canvas, fg_color="#000000")
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
            face_frame = ctk.CTkFrame(frame, fg_color="#1C2526", corner_radius=5)
            label = ctk.CTkLabel(face_frame, image=imgtk, text="")
            label.pack(pady=5)
            ctk.CTkLabel(face_frame, text=name, font=("Courier New", 12), text_color="#00FF00").pack()
            ctk.CTkButton(face_frame, text="> Delete", command=lambda f=fname: delete_face(f, main_frame), font=("Courier New", 12), fg_color="#00FFFF", hover_color="#00FF00", text_color="#000000").pack(pady=5)
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
    canvas = ctk.CTkCanvas(main_frame, bg="#000000", highlightthickness=0)
    scroll = ctk.CTkScrollbar(main_frame, orientation="vertical", command=canvas.yview, fg_color="#1C2526")
    frame = ctk.CTkFrame(canvas, fg_color="#000000")
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
            face_frame = ctk.CTkFrame(frame, fg_color="#1C2526", corner_radius=5)
            label = ctk.CTkLabel(face_frame, image=imgtk, text="")
            label.pack(pady=5)
            entry = ctk.CTkEntry(face_frame, placeholder_text="Enter name", font=("Courier New", 12), text_color="#00FF00", fg_color="#1C2526")
            entry.pack(pady=5)
            ctk.CTkButton(face_frame, text="> Register",
                          command=lambda f=fname, e=entry: register_unknown(f, e, main_frame, status_label),
                          font=("Courier New", 12), fg_color="#00FFFF", hover_color="#00FF00", text_color="#000000").pack(pady=5)
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
    status_label.configure(text=f"> Registered {name} from unknown")
    messagebox.showinfo("Success", f"{name} registered!")
    review_unknown_faces(status_label, main_frame)

def show_reports(main_frame):
    clear_main_frame(main_frame)
    tree = tk.ttk.Treeview(main_frame, columns=("Time", "Object", "Confidence", "Name"), show="headings")
    tree.configure(style="Hacker.Treeview")
    for col in ("Time", "Object", "Confidence", "Name"):
        tree.heading(col, text=col)
        tree.column(col, width=150)
    cursor.execute("SELECT timestamp, detected_object, confidence, face_name FROM detections ORDER BY timestamp DESC LIMIT 100")
    for row in cursor.fetchall():
        tree.insert("", "end", values=row)
    tree.pack(fill="both", expand=True)

def show_settings(status_label, main_frame):
    clear_main_frame(main_frame)
    ctk.CTkLabel(main_frame, text="> System Settings", font=("Courier New", 16, "bold"), text_color="#00FF00").pack(pady=10)
    ctk.CTkLabel(main_frame, text="> Theme:", font=("Courier New", 12), text_color="#00FF00").pack(pady=5)
    theme_option = ctk.CTkOptionMenu(main_frame, values=["Dark", "Light", "System"], command=lambda mode: ctk.set_appearance_mode(mode),
                                     font=("Courier New", 12), fg_color="#00FFFF", button_color="#00FFFF", button_hover_color="#00FF00", text_color="#000000")
    theme_option.set("Dark")
    theme_option.pack(pady=5)

def clear_main_frame(main_frame):
    for widget in main_frame.winfo_children():
        widget.destroy()

# Pulse animation for alert label
def pulse_alert(alert_label):
    def update_opacity():
        if alert_label.cget("text"):
            current_alpha = float(alert_label.cget("text_color")[1:7]) / 255.0
            new_alpha = 1.0 if current_alpha < 0.5 else 0.5
            alert_label.configure(text_color=f"#FF0000{int(new_alpha * 255):02x}")
            alert_label.after(500, update_opacity)
    update_opacity()

# Main GUI with hackers template
def main_ui():
    global detection_thread
    detection_thread = None
    root = ctk.CTk()
    root.title("HACKSEC v1.0")
    root.geometry("900x600")
    root.configure(fg_color="#000000")

    # Custom Treeview style for reports
    style = tk.ttk.Style()
    style.configure("Hacker.Treeview", background="#000000", foreground="#00FF00", fieldbackground="#000000", font=("Courier New", 10))
    style.map("Hacker.Treeview", background=[("selected", "#1C2526")])

    # Sidebar
    sidebar = ctk.CTkFrame(root, width=200, corner_radius=0, fg_color="#000000", border_color="#00FF00", border_width=2)
    sidebar.pack(side="left", fill="y")
    ctk.CTkLabel(sidebar, text="> HACKSEC CORE", font=("Courier New", 16, "bold"), text_color="#00FF00").pack(pady=20)

    buttons = [
        ("Register Member", lambda: register_face(status_label, main_frame), "> Register a new entity"),
        ("Start Monitoring", lambda: start_detection(status_label, alert_label, main_frame), "> Initiate surveillance"),
        ("Family Member", lambda: show_faces(main_frame), "> View authorized entities"),
        ("Review Unknown", lambda: review_unknown_faces(status_label, main_frame), "> Analyze unknown entities"),
        ("Reports", lambda: show_reports(main_frame), "> Access detection logs"),
        ("Settings", lambda: show_settings(status_label, main_frame), "> Configure system")
    ]

    for text, command, tooltip in buttons:
        btn = ctk.CTkButton(sidebar, text=f"> {text}", command=command, font=("Courier New", 12), fg_color="#00FFFF", hover_color="#00FF00", text_color="#000000", height=40,
                            corner_radius=5, border_color="#00FF00", border_width=1)
        btn.pack(pady=5, padx=10, fill="x")
        btn.bind("<Enter>", lambda e, t=tooltip: status_label.configure(text=t))
        btn.bind("<Leave>", lambda e: status_label.configure(text="> Idle"))

    # Main Frame
    main_frame = ctk.CTkFrame(root, corner_radius=10, fg_color="#1C2526", border_color="#00FFFF", border_width=2)
    main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    ctk.CTkLabel(main_frame, text="> HACKSEC v1.0", font=("Courier New", 20, "bold"), text_color="#00FF00").pack(pady=20)
    ctk.CTkLabel(main_frame, text="> Select a directive from the core.", font=("Courier New", 14), text_color="#00FFFF").pack(pady=10)

    # Status Bar
    status_frame = ctk.CTkFrame(root, height=30, corner_radius=0, fg_color="#000000", border_color="#00FF00", border_width=2)
    status_frame.pack(side="bottom", fill="x")
    status_label = ctk.CTkLabel(status_frame, text="> Idle", font=("Courier New", 12), text_color="#00FF00")
    status_label.pack(side="left", padx=10)
    alert_label = ctk.CTkLabel(status_frame, text="", font=("Courier New", 12), text_color="#FF0000")
    alert_label.pack(side="right", padx=10)

    # Start pulse animation for alerts
    pulse_alert(alert_label)

    root.protocol("WM_DELETE_WINDOW", lambda: cleanup(root))
    root.mainloop()

def cleanup(root):
    global detection_thread
    if detection_thread:
        detection_thread.stop()
    conn.close()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    root.quit()

if __name__ == "__main__":
    main_ui()