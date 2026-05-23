import json
from pathlib import Path

import pandas as pd
import torch

from train_titanic import METADATA_FILE
from train_titanic import MODEL_FILE
from train_titanic import TEST_FILE
from train_titanic import Model
from train_titanic import build_features


DATA_DIR = Path(__file__).resolve().parent / "titanic"
TRAIN_FILE = DATA_DIR / "train.csv"
OUTPUT_FILE = DATA_DIR / "submission.csv"


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    _, test_features, metadata = build_features(train_df, test_df)

    saved_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    feature_columns = saved_metadata["feature_columns"]

    missing_columns = [column for column in feature_columns if column not in test_features.columns]
    for column in missing_columns:
        test_features[column] = 0.0
    test_features = test_features.reindex(columns=["dataset", "PassengerId", "Survived", *feature_columns], fill_value=0.0)

    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    model = Model(len(feature_columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = torch.tensor(test_features[feature_columns].values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int().view(-1).tolist()

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions,
        }
    )
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"feature count={len(feature_columns)}")
    print(submission.head())
    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
