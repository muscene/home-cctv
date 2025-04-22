import os

# Directories for face data
REGISTERED_FACES_DIR = "registered_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Database
DB_FILE = "detection_records.db"

# Email Configuration
EMAIL_SENDER = "iot@vrt.rw"
EMAIL_PASSWORD = "TheGreat@123"
EMAIL_RECIPIENT = "gasasiras103@gmail.com"
SMTP_SERVER = "mail.vrt.rw"
SMTP_PORT = 465

# PaddlePaddle Model Directory
MASK_MODEL_DIR = "models/paddle"

# YOLO Weapon Detection Model
WEAPON_MODEL_PATH = "yolov5nu.pt"
