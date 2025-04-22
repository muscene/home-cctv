import cv2
import numpy as np
import time
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

# Anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (255, 0, 0))

def getOutputsNames(net):
    """Get the names of the output layers of the network."""
    layersNames = net.getLayerNames()
    unconnectedOutLayers = net.getUnconnectedOutLayers()
    if isinstance(unconnectedOutLayers, (int, np.integer)):
        return [layersNames[unconnectedOutLayers - 1]]
    else:
        return [layersNames[i - 1] for i in unconnectedOutLayers]

def inference(net, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160)):
    """Run inference on an image and draw the detection results."""
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net.setInput(blob)
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))

    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]

    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh, iou_thresh)

    mask_detected = False

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]

        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], 2)
        label = f"{id2class[class_id]}: {conf:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id], 2)

        if id2class[class_id] == 'Mask':
            mask_detected = True

    return image, mask_detected

def run_on_webcam(net, conf_thresh=0.5):
    cap = cv2.VideoCapture(0)  # Use the default webcam

    if not cap.isOpened():
        raise ValueError("Error: Cannot open webcam.")

    print("Press 'ESC' to exit the webcam feed.")

    recording = False
    start_time = None
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame, mask_detected = inference(net, frame, target_shape=(260, 260), conf_thresh=conf_thresh)

        if mask_detected and not recording:
            # Start recording
            recording = True
            start_time = time.time()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('mask_detected.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print("Recording started...")

        if recording:
            out.write(frame)
            if time.time() - start_time >= 5:
                # Stop recording after 5 seconds
                recording = False
                out.release()
                print("Recording saved as 'mask_detected.avi'.")

        cv2.imshow('Face Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'models/face_mask_detection.caffemodel'
    proto_path = 'models/face_mask_detection.prototxt'

    net = cv2.dnn.readNet(model_path, proto_path)
    run_on_webcam(net, conf_thresh=0.5)
