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
    df = pd.read_csv(label_features_path,nrows=1000)
    logger.info(f"Loaded {len(df)} rows from {label_features_path}.")

    training_data = []
    for row in tqdm(df.iterrows(), total=1000, desc="Training model"):
        training_data.append(row)
    logger.success("Modeling training complete.")
    print(training_data)
    # -----------------------------------------


if __name__ == "__main__":
    app()
