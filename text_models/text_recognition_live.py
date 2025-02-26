import json
import cv2
import time
import pyttsx3
from paddleocr import PaddleOCR
import os

# Get the absolute path of the JSON file
json_path = os.path.join(os.path.dirname(__file__), "best_text_ocr_params.json")

with open(json_path, "r") as f:
    best_text_ocr_params = json.load(f)
# Initialize PaddleOCR with best hyperparameters
ocr = PaddleOCR(**best_text_ocr_params)

def adjust_brightness_contrast(img, alpha=1.5, beta=20):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def perform_live_text():
    engine = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    last_speech_time = 0
    speech_gap = 2  # Min time between speech outputs

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Adjust brightness and contrast
        adjusted_frame = adjust_brightness_contrast(frame)

        # Perform OCR
        result = ocr.ocr(adjusted_frame, cls=False)
        detected_text = "\n".join([line[1][0] for line in result[0] if line[1][1] > best_text_ocr_params["drop_score"]])

        # Display detected text on screen
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, detected_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Speak detected text if enough time has passed
        if detected_text and (time.time() - last_speech_time) > speech_gap:
            engine.say(detected_text)
            engine.runAndWait()
            last_speech_time = time.time()

        cv2.imshow('Live OCR - Press C to Stop', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()
