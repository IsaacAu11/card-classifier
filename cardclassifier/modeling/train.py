from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd


from cardclassifier.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    label_features_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")

    training_data = training_data_generator(label_features_path)

    
    

    logger.success("Modeling training complete.")
    # -----------------------------------------

def training_data_generator(label_features_path):
    df = pd.read_csv(label_features_path, nrows=1000)
    
    # Reorder columns to place 'class_name' as the second column
    columns = ['image_data', 'class_name'] + [col for col in df.columns if col not in ['image_data', 'class_name']]
    df = df[columns]

    training_data = []

    for row in tqdm(df.iterrows(), total=len(df), desc="Training model"):
        training_data.append(row)
        print(row)
    return training_data

if __name__ == "__main__":
    app()
