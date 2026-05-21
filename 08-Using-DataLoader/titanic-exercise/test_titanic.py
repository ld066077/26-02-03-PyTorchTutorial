from pathlib import Path

import pandas as pd
import torch

from train_titanic import FEATURE_COLUMNS
from train_titanic import Model
from train_titanic import preprocess_dataframe


DATA_DIR = Path(__file__).resolve().parent / "titanic"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = DATA_DIR / "titanic_model.pt"
OUTPUT_FILE = DATA_DIR / "submission.csv"


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    model = Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(TEST_FILE)
    processed_df = preprocess_dataframe(df)
    features = torch.tensor(processed_df[FEATURE_COLUMNS].values, dtype=torch.float32)

    with torch.no_grad():
        probabilities = model(features)
        predictions = (probabilities >= 0.5).int().view(-1).tolist()

    submission = pd.DataFrame(
        {
            "PassengerId": df["PassengerId"],
            "Survived": predictions,
        }
    )
    submission.to_csv(OUTPUT_FILE, index=False)
    print(submission.head())
    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
