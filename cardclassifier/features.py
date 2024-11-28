from pathlib import Path

import typer
from loguru import logger
import keras
import pandas as pd

from cardclassifier.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "train",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    pass

if __name__ == "__main__":
    app()
