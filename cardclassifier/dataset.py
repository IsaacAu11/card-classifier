from pathlib import Path

import numpy as np
import typer
from loguru import logger
import os
from PIL import Image
import pandas as pd
import base64
from io import BytesIO

from cardclassifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "train",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    
    process_data_to_csv(input_path, output_path)

    logger.success("Processing dataset complete.")
    # -----------------------------------------

def process_data_to_csv(data_dir, output_csv):
    # List to hold the data
    data = []

    # Traverse the directory
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):  # Check if it's a directory
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                if os.path.isfile(file_path):  # Check if it's a file
                    # Read the image
                    img = Image.open(file_path).convert('L')
                    img = img.resize((224, 224))  # Resize to a fixed size if needed
                    img = np.array(img)/255
                    img_array = np.array(img)  # Convert to numpy array

                    # Option 1: Flatten the image array
                    img_data = img_array.flatten().tolist()  # Flatten and convert to list

                    # Append class name and image data
                    data.append({'class_name': class_name, 'image_data': img_data})

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    app()
