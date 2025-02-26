import cv2
import torch
import json
import os
from gtts import gTTS
from datetime import datetime

def adjust_brightness_contrast(img, alpha=1.5, beta=20):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def perform_object_detection():
    cap = cv2.VideoCapture(0)

    # Load best hyperparameters
    json_path = os.path.join(os.path.dirname(__file__), "best_object_detection_params.json")
    with open(json_path, "r") as f:
        best_params = json.load(f)

    # Load YOLO model with best parameters
    model = torch.hub.load('ultralytics/yolov5', best_params["model"], pretrained=True)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Press "c" to Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img_path = 'static/captured/captured_image.jpg'
            cv2.imwrite(img_path, frame)
            print("Image captured and saved.")

            adjusted_frame = adjust_brightness_contrast(frame)
            adjusted_img_path = 'static/captured/adjusted_captured_image.jpg'
            cv2.imwrite(adjusted_img_path, adjusted_frame)

            # Perform object detection
            results = model(adjusted_frame, 
                            imgsz=best_params["img_size"], 
                            conf=best_params["conf_thres"], 
                            iou=best_params["iou_thres"], 
                            half=best_params["half"], 
                            augment=best_params["augment"])

            # Ensure there are detections
            detected_objects = []
            if results.xyxy[0].shape[0] > 0:  # Check if any object detected
                detected_objects = [model.names[int(x)] for x in results.xyxy[0][:, -1].tolist()]  # Convert tensor to list

            description_text = f"The objects detected are: {', '.join(detected_objects)}." if detected_objects else "No objects detected."

            # Generate audio
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            audio_path = os.path.join('static/audio', f'description_{timestamp}.mp3')
            tts = gTTS(description_text, lang='en')
            tts.save(audio_path)

            cap.release()
            cv2.destroyAllWindows()
            return description_text, os.path.basename(adjusted_img_path), os.path.basename(audio_path)

        elif key == 27:  # Press 'Esc' to exit
            cap.release()
            cv2.destroyAllWindows()
            break
