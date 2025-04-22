import cv2
from ultralytics import YOLO

# Load the custom YOLOv5 model from a local file
weapon_model = YOLO('yolov10n.pt')  # Adjust the path if necessary

# Initialize the Pi Camera or other connected camera using OpenCV
cap = cv2.VideoCapture(1)  # 0 is typically the default camera (use 1 or 2 for other cameras)

# Set the camera resolution (optional, can be adjusted for performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform weapon detection on the frame
    results = weapon_model(frame)  # YOLO automatically handles the conversion to RGB

    # Access the first result (YOLOv5 returns a list of results)
    result = results[0]

    # Draw bounding boxes on the frame manually (optional)
    annotated_frame = result.plot()  # Get annotated image with bounding boxes

    # Display the frame with the bounding boxes
    cv2.imshow("Weapon Detection", annotated_frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
