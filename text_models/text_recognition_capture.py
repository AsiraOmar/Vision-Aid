import json
import os
import cv2
from paddleocr import PaddleOCR
from gtts import gTTS
from datetime import datetime


# Get the absolute path of the JSON file
json_path = os.path.join(os.path.dirname(__file__), "best_text_ocr_params.json")

with open(json_path, "r") as f:
    best_text_ocr_params = json.load(f)

# Initialize PaddleOCR with best hyperparameters
ocr = PaddleOCR(**best_text_ocr_params)

def adjust_brightness_contrast(img, alpha=1.5, beta=20):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def perform_text_capture(output_folder="static"):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press "c" to Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == 27:  # Press 'Esc' to exit
            return "Capture cancelled", "", ""

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    img_path = os.path.join(output_folder, "captured", f'captured_image_{timestamp}.jpg')
    cv2.imwrite(img_path, frame)

    # Adjust brightness and contrast
    adjusted_frame = adjust_brightness_contrast(frame)
    adjusted_img_path = os.path.join(output_folder, "captured", f'adjusted_image_{timestamp}.jpg')
    cv2.imwrite(adjusted_img_path, adjusted_frame)

    # Perform OCR
    result = ocr.ocr(adjusted_frame, cls=False)
    text_output = "\n".join([line[1][0] for line in result[0] if line[1][1] > best_text_ocr_params["drop_score"]])

    if not text_output.strip():
        return "No text detected", "", ""

    # Convert text to speech
    tts = gTTS(text=text_output, lang='en')
    audio_path = os.path.join(output_folder, "audio", f'output_audio_{timestamp}.mp3')
    tts.save(audio_path)

    return text_output, adjusted_img_path, audio_path, timestamp
