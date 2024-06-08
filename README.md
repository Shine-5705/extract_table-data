
# OCR Table Extraction

This project uses PaddleOCR to extract tabular data from images and save the extracted data into separate CSV files. The script processes each image in a specified folder, performs OCR, and aligns the detected text into rows to form a table, which is then saved as a CSV file.

## Features

- Automated OCR Processing: Automatically processes all images in a specified folder.
- Table Alignment: Extracted text is aligned into rows to form tables.
- CSV Output: Saves the extracted table data into separate CSV files named after the image files.

## Requirements

- Python 3.6+
- PaddleOCR
- OpenCV
- Pillow
- Matplotlib
- Pandas

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Shine-5705/extract_table-data.git
    cd ocr-table-extraction
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv myvenv
    source myvenv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:

    ```sh
    pip install opencv-python-headless paddlepaddle paddleocr matplotlib pandas
    ```

## Usage

1. Place your images in a directory (e.g., input_images).

2. Modify the `image_dir` and `csv_output_dir` variables in the script to point to your image directory and desired output directory for CSV files:

    ```python
    image_dir = '[Business Quant] Image dataset for OCR'
    csv_output_dir = 'extracted'
    ```

3. Run the script:

    ```sh
    python main.py
    ```

## Script

Here's a brief overview of what the script does:

- Initialization: Sets up PaddleOCR and the directories for images and CSV output.
- Image Processing: Loops through each image in the specified folder, performs OCR, and extracts text data.
- Data Alignment: Aligns the extracted text into rows to form a table.
- CSV Saving: Saves the aligned table data into a CSV file named after the image file.

```python
from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.font_manager as fm

# Initialize PaddleOCR with the appropriate model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Directory containing the images
image_dir = '/path/to/your/image/folder'

# Directory to save the CSV files
csv_output_dir = '/path/to/save/csv/files'
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
