# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:22:38 2025

@author: asira
"""

import time
import json
import cv2
import numpy as np
import glob
import os
from paddleocr import PaddleOCR, draw_ocr

# Define hyperparameter search space
hyperparams = [
    {"det_db_box_thresh": 0.3, "det_db_unclip_ratio": 1.8, "use_angle_cls": False, "rec_algorithm": "SVTR_LCNet",
     "rec_image_shape": "3, 48, 320", "rec_char_type": "en", "use_direction_classify": False, "drop_score": 0.5},
    
    {"det_db_box_thresh": 0.2, "det_db_unclip_ratio": 1.6, "use_angle_cls": False, "rec_algorithm": "CRNN",
     "rec_image_shape": "3, 32, 100", "rec_char_type": "en", "use_direction_classify": True, "drop_score": 0.6},
    
    {"det_db_box_thresh": 0.25, "det_db_unclip_ratio": 1.9, "use_angle_cls": True, "rec_algorithm": "SVTR_LCNet",
     "rec_image_shape": "3, 48, 320", "rec_char_type": "en", "use_direction_classify": True, "drop_score": 0.5}
]

best_time = float("inf")
best_params = None
image_folder = "C:/Users/asira/Downloads/WeSee-main (1)/WeSee-main/text_models/text_images/"  # Image folder

# Load all JPG and JPEG images
image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.jpeg"))

# Ensure images exist
if not image_paths:
    print("No images found in the folder!")
    exit()

# Iterate through hyperparameter sets
for params in hyperparams:
    total_time = 0

    # Initialize OCR model with current hyperparameters
    ocr_reader = PaddleOCR(
        det_db_box_thresh=params["det_db_box_thresh"],
        det_db_unclip_ratio=params["det_db_unclip_ratio"],
        use_angle_cls=params["use_angle_cls"],
        rec_algorithm=params["rec_algorithm"],
        rec_image_shape=params["rec_image_shape"],
        rec_char_type=params["rec_char_type"],
        use_direction_classify=params["use_direction_classify"],
        drop_score=params["drop_score"]
    )

    for image_path in image_paths:
        start_time = time.time()
        result = ocr_reader.ocr(image_path)  # Process image
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

    avg_time = total_time / len(image_paths)  # Average inference time

    if avg_time < best_time:
        best_time = avg_time
        best_params = params
        best_reader = ocr_reader  # Store best OCR model

# Save best hyperparameters
with open("best_text_ocr_params.json", "w") as f:
    json.dump(best_params, f)

print(f"Best Hyperparameters: {best_params}")

# Create output folder
output_folder = "C:/Users/asira/Downloads/ocr_results/"
os.makedirs(output_folder, exist_ok=True)

# Font path for visualization
font_path = "C:/Users/asira/anaconda3/Lib/site-packages/paddleocr/doc/fonts/simfang.ttf"
if not os.path.exists(font_path):
    print(f"Font file not found: {font_path}. Using Arial as fallback.")
    font_path = "C:/Windows/Fonts/Arial.ttf"

# Process and save results for each image
for image_path in image_paths:
    image = cv2.imread(image_path)
    result = best_reader.ocr(image_path)  # Use best model
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    visualized_image = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, visualized_image)
    print(f"Saved OCR-processed image: {output_path}")
