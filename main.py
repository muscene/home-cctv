from flask import Flask, render_template, Response, redirect, url_for, send_from_directory,jsonify
import cv2
import sqlite3
import io
import base64
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# Directories for saved images
REGISTERED_FACES_DIR = "registered_faces"
UNKNOWN_FACES_DIR = "unknown_faces"

# Global variable to control detection loop
is_running = False
camera = cv2.VideoCapture(1)

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

# Route to display the home page
@app.route('/')
def index():
    registered_images = os.listdir(REGISTERED_FACES_DIR)
    unknown_images = os.listdir(UNKNOWN_FACES_DIR)
    plot_url = generate_plot()
    return render_template('index.html', 
                           registered_images=registered_images, 
                            plot_url = plot_url,
                           unknown_images=unknown_images,
                       
                           is_running=is_running)
@app.route('/unknown')
def unknown():
    unknown_images = os.listdir(UNKNOWN_FACES_DIR)
    return render_template('unknown.html',unknown_images=unknown_images,
                           is_running=is_running)
@app.route('/registered')
def registered():
    registered_images = os.listdir(REGISTERED_FACES_DIR)
   
    return render_template('registered.html', 
                           registered_images=registered_images, 
                           is_running=is_running)
# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start the detection loop
@app.route('/start_detection')
def start_detection():
    global is_running
    is_running = True
    return redirect(url_for('index'))

# Route to stop the detection loop
@app.route('/stop_detection')
def stop_detection():
    global is_running
    is_running = False
    return redirect(url_for('index'))

# Route to serve registered face images
@app.route('/registered_faces/<filename>')
def registered_faces(filename):
    return send_from_directory(REGISTERED_FACES_DIR, filename)

# Route to serve unknown face images
@app.route('/unknown_faces/<filename>')
def unknown_faces(filename):
    return send_from_directory(UNKNOWN_FACES_DIR, filename)

# Function to generate the plot
# Function to generate the plot
def generate_plot():
    try:
        # Connect to the database
        conn = sqlite3.connect('detection_records.db')
        cursor = conn.cursor()
        cursor.execute('SELECT detected_object FROM detections')
        data = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()

    # Process data for plotting
    if not data:
        print("No data available for plotting.")
        return None

    objects = [row[0] for row in data]
    object_counts = {}
    for obj in objects:
        object_counts[obj] = object_counts.get(obj, 0) + 1

    # Plot the data
    plt.figure(figsize=(8,4))
    plt.bar(object_counts.keys(), object_counts.values(), color='skyblue')
    plt.xlabel('Detected Objects')
    plt.ylabel('Count')
    plt.title('Detection Log Summary')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image to base64
    return base64.b64encode(img.getvalue()).decode()
@app.route('/plot')
def plot():
    plot_base64 = generate_plot()
    if plot_base64 is None:
        return "No data available for plotting or an error occurred.", 500
    
    return f'<img src="data:image/png;base64,{plot_base64}" alt="Plot">'


def get_detection_data():
    """
    Fetches detection data from the 'detections' table in the database.

    Returns:
        list: A list of dictionaries containing detection records.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect('detection_records.db')
        cursor = conn.cursor()
        
        # Query to retrieve data
        cursor.execute('SELECT id, detected_object, timestamp FROM detections')
        rows = cursor.fetchall()
        
        # Check if there's data
        if not rows:
            print("No detection records found in the database.")
            return []
        
        # Format the data into a list of dictionaries
        data = []
        for row in rows:
            data.append({
                'id': row[0],
                'detected_object': row[1],
                'timestamp': row[2]
            })

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

    finally:
        if conn:
            conn.close()

    return data
@app.route('/api/detections', methods=['GET'])
def detections():
    data = get_detection_data()
    return jsonify(data)
@app.route('/logs')
def logs():
    logs = get_detection_data()
    return render_template('logs.html', logs=logs, current_year=2024)
# Main function to run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
