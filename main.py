import os
from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

# Initialize PaddleOCR with the appropriate model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Directory containing the images
image_dir = '[Business Quant] Image dataset for OCR'

# Directory to save the CSV files
csv_output_dir = 'extracted'
os.makedirs(csv_output_dir, exist_ok=True)

# Function to process a single image
def process_image(image_path, csv_output_path):
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Perform OCR on the image
    result = ocr.ocr(image, cls=True)

    # Extract the text and bounding boxes
    data = []
    for line in result:
        for word_info in line:
            bbox, (text, score) = word_info
            data.append((text, bbox, score))

    # Sort the data by the top-left y-coordinate of the bounding boxes to align text into rows
    data.sort(key=lambda x: (x[1][0][1], x[1][0][0]))

    # Post-process the data to align into a table
    rows = []
    current_row = []
    current_y = data[0][1][0][1]

    for text, bbox, score in data:
        bbox_top_left_y = bbox[0][1]
        if bbox_top_left_y - current_y > 10:  # Adjust the threshold as needed
            rows.append(current_row)
            current_row = [text]
            current_y = bbox_top_left_y
        else:
            current_row.append(text)

    rows.append(current_row)  # Append the last row

    # Convert rows into a DataFrame for better visualization and alignment
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_output_path, index=False)
    print(f"Saved CSV to {csv_output_path}")

# Loop through each image in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(image_dir, filename)
        csv_output_path = os.path.join(csv_output_dir, f"{os.path.splitext(filename)[0]}.csv")
        process_image(image_path, csv_output_path)
