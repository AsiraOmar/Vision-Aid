import cv2
import torch
import pyttsx3
import os
import time
from ultralytics import YOLO

def object_detection():
    """Perform real-time object detection and generate scene descriptions."""
    
    # Load YOLOv8 model
    model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    model = YOLO(model_path)
    tts_engine = pyttsx3.init()

    def generate_scene_description(objects):
        """Generate a description of the detected scene."""
        return f"{' and '.join(objects)} are visible in the scene." if objects else "No objects detected."

    cap = cv2.VideoCapture(0)
    last_speech_time = time.time() - 6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        detected_objects = []
        for result in results:
            detected_objects.extend(result.names[int(box[5])] for box in result.boxes.xyxy)

        # Speak detected objects every 6 seconds
        if detected_objects and (time.time() - last_speech_time) >= 6:
            last_speech_time = time.time()
            scene_description = generate_scene_description(detected_objects)
            print(scene_description)
            tts_engine.say(scene_description)
            tts_engine.runAndWait()

        # Display the results
        annotated_frame = results[0].plot()
        cv2.imshow('Press "c" to stop detection.', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected_objects
