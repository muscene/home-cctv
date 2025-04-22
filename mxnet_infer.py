from flask import Flask, render_template, Response, redirect, url_for, send_from_directory
import cv2
import os
import re

app = Flask(__name__)

# Directories for saved images
REGISTERED_FACES_DIR = "registered_faces"
UNKNOWN_FACES_DIR = "unknown_faces"

# Global variable to control detection loop
is_running = False
camera = cv2.VideoCapture(0)

# Function to generate video frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Group registered face images by name prefix (excluding trailing numbers)
def get_grouped_registered_faces():
    images = os.listdir(REGISTERED_FACES_DIR)
    grouped_images = {}
    
    for image in images:
        match = re.match(r"(.+?)_\d+\.(jpg|png)", image)
        if match:
            name_prefix = match.group(1)
            if name_prefix not in grouped_images:
                grouped_images[name_prefix] = image
        else:
            # If no trailing number, add the image directly
            name_prefix = os.path.splitext(image)[0]
            if name_prefix not in grouped_images:
                grouped_images[name_prefix] = image

    return grouped_images.values()

# Route to display the home page
@app.route('/')
def index():
    registered_images = get_grouped_registered_faces()
    unknown_images = os.listdir(UNKNOWN_FACES_DIR)
    return render_template('index.html', 
                           registered_images=registered_images, 
                           unknown_images=unknown_images)

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve registered face images
@app.route('/registered_faces/<filename>')
def registered_faces(filename):
    return send_from_directory(REGISTERED_FACES_DIR, filename)

# Route to serve unknown face images
@app.route('/unknown_faces/<filename>')
def unknown_faces(filename):
    return send_from_directory(UNKNOWN_FACES_DIR, filename)

# Main function to run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

