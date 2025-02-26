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

def perform_ocr_and_audio(image_path, output_folder="static"):
    img = cv2.imread(image_path)

    # Adjust brightness and contrast
    img_adjusted = adjust_brightness_contrast(img)

    # Save adjusted image
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    adjusted_image_path = os.path.join(output_folder, "captured", f'adjusted_image_{timestamp}.jpg')
    cv2.imwrite(adjusted_image_path, img_adjusted)

    # Perform OCR
    result = ocr.ocr(img_adjusted, cls=False)
    text = "\n".join([line[1][0] for line in result[0] if line[1][1] > best_text_ocr_params["drop_score"]])

    if not text.strip():
        return "No text detected", "", ""

    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    audio_path = os.path.join(output_folder, "audio", f'output_{timestamp}.mp3')
    tts.save(audio_path)

    return text, audio_path, adjusted_image_path, timestamp
