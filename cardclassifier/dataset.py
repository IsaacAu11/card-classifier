from pathlib import Path

import numpy as np
import typer
from loguru import logger
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

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
    logger.info("Starting dataset processing...")
    
    process_data_to_csv(input_path, output_path)

    logger.success("Processing dataset complete.")
    # -----------------------------------------

def process_data_to_csv(data_dir, output_csv):
    # List to hold the data
    data = []

    # Traverse the directory
    class_names = os.listdir(data_dir)
    logger.info(f"Found {len(class_names)} classes in the dataset.")

    # Use tqdm to create a progress bar for class processing
    for class_name in tqdm(class_names, desc="Processing classes", unit="class"):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):  # Check if it's a directory
            # Use tqdm to create a progress bar for file processing
            for filename in tqdm(os.listdir(class_path), desc=f"Processing files in {class_name}", unit="file"):
                file_path = os.path.join(class_path, filename)
                if os.path.isfile(file_path):  # Check if it's a file
                    # Read the image
                    img = Image.open(file_path).convert('L')
                    img = img.resize((224, 224))  # Resize to a fixed size if needed
                    img = np.array(img) / 255
                    img_array = np.array(img)  # Convert to numpy array

                    # Option 1: Flatten the image array
                    img_data = img_array.flatten().tolist()  # Flatten and convert to list

                    # Append class name and image data
                    data.append({'class_name': class_name, 'image_data': img_data})

                    logger.info(f"Processed file: {filename} in class: {class_name}")

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.success(f"Dataset saved to {output_csv}")

if __name__ == "__main__":
    app()
