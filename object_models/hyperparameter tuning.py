# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:31:02 2025

@author: asira
"""

import time
import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Define an extensive hyperparameter search space
hyperparams = [
    {"model": "yolov8n.pt", "img_size": 416, "conf_thres": 0.25, "iou_thres": 0.45, "half": False, "augment": False},
    {"model": "yolov8s.pt", "img_size": 512, "conf_thres": 0.3, "iou_thres": 0.5, "half": True, "augment": False},
    {"model": "yolov8m.pt", "img_size": 640, "conf_thres": 0.5, "iou_thres": 0.5, "half": True, "augment": True},
    {"model": "yolov8l.pt", "img_size": 960, "conf_thres": 0.5, "iou_thres": 0.45, "half": True, "augment": True},
    ]

best_time = float("inf")
best_params = None
image_path = "C:/Users/asira/Downloads/test2.jpg"
image = cv2.imread(image_path)

for params in hyperparams:
    model = YOLO(params["model"])
    start_time = time.time()
    
    results = model(image, 
                    imgsz=params["img_size"], 
                    conf=params["conf_thres"], 
                    iou=params["iou_thres"], 
                    half=params["half"], 
                    augment=params["augment"])
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time < best_time:
        best_time = elapsed_time
        best_params = params
        best_results = results  # Store the best results for visualization

# Save best hyperparameters
with open("best_object_detection_params.json", "w") as f:
    json.dump(best_params, f)

print(f"Best Hyperparameters: {best_params}")

# Save detected image
if best_results:
    for result in best_results:
        detected_image = result.plot()  # Get the image with bounding boxes
        output_path = "C:/Users/asira/Downloads/detected_image.jpg"  # Save path
        cv2.imwrite(output_path, detected_image)
        print(f"Detected image saved to: {output_path}")
