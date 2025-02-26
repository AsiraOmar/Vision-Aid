# Vision Aid (WeSee) Project

## Overview
Vision Aid is a real-time object detection and text recognition system designed to assist visually impaired individuals. The project integrates YOLOv8 for object detection and PaddleOCR for text recognition into a Flask-based web application, allowing users to capture, upload, or use live detection to recognize objects and text in their environment.

## Features
- **Real-time Object Detection**: Detect objects from a live camera feed.
- **Text Recognition (OCR)**: Extract and recognize text from images.
- **Multiple Input Methods**:
  - Capture an image from a webcam.
  - Upload an image for detection.
  - Perform live detection using a connected camera.
- **Hyperparameter Tuning**:
  - Object detection models use optimized parameters from `best_object_detection_params.json`.
  - Text recognition models use optimized parameters from `best_text_ocr_params.json`.

## Project Structure
```
wesee/
│── object_models/       # Contains object detection models (YOLOv8)
│── text_models/         # Contains text recognition models (PaddleOCR)
│── templates/           # HTML templates for the Flask web app
│── static/              # Static assets (CSS, JavaScript, images)
│── main.py              # Flask application entry point
│── best_object_detection_params.json  # Optimized YOLOv8 parameters
│── best_text_ocr_params.json          # Optimized PaddleOCR parameters
│── requirements.txt      # Dependencies
│── README.md            # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- Pip
- Virtual Environment (recommended)

### Steps
1. Clone the repository:

2. Create and activate a virtual environment:

3. Install dependencies:


## Usage
1. Start the Flask web application:
   ```bash
   python main.py
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
3. Use the web interface to:
   - Capture an image using the webcam.
   - Upload an image for analysis.
   - Start live detection.

## Model Details
### Object Detection (YOLOv8)
- Uses yolov8 models.
- Optimized with the best hyperparameters from `best_object_detection_params.json`.
- Processes images for object localization and classification.

### Text Recognition (PaddleOCR)
- Configured with `SVTR_LCNet` for text recognition.
- Supports English character recognition.
- Uses tuned hyperparameters from `best_text_ocr_params.json`.

## Evaluation
- **Object Detection**: Evaluated on a test dataset with performance metrics (mAP, accuracy).
- **Text Recognition**: Evaluated only on images with detected text, using batch processing for large datasets.

## Future Improvements
- Enhance support for additional languages in text recognition.
- Improve real-time performance with hardware acceleration.
- Develop a mobile-friendly version.

## License
This project is licensed under the MIT License.


