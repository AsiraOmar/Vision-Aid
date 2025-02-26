import json
import os
from gtts import gTTS
from PIL import Image
from datetime import datetime
from flask import Flask
import torch
from ultralytics import YOLO  # Correct import for YOLOv8

app = Flask(__name__)
app.config['AUDIO_FOLDER'] = 'static/audio'

# Load best hyperparameters
json_path = os.path.join(os.path.dirname(__file__), "best_object_detection_params.json")
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        best_params = json.load(f)
else:
    raise FileNotFoundError(f"Hyperparameter file {json_path} not found!")

# Load YOLOv8 model
model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")  # Ensure correct path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

try:
    model = YOLO(model_path)  # Load YOLOv8 model
except Exception as e:
    raise RuntimeError(f"Error loading YOLOv8 model: {e}")

def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Perform detection with best parameters
    results = model.predict(
        source=img, 
        imgsz=best_params.get("img_size", 640), 
        conf=best_params.get("conf_thres", 0.25), 
        iou=best_params.get("iou_thres", 0.45)
    )

    # Extract detected objects
    detected_objects = set()  # Use set to avoid duplicates
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            detected_objects.add(cls)

    description_text = f"The objects detected are: {', '.join(detected_objects)}." if detected_objects else "No objects detected."

    # Generate audio
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], f'description_{timestamp}.mp3')

    tts = gTTS(description_text, lang='en')
    tts.save(audio_path)

    return description_text, audio_path, timestamp
