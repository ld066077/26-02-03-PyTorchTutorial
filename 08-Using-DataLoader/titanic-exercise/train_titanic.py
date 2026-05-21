from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).resolve().parent / "titanic"
TRAIN_FILE = DATA_DIR / "train.csv"
MODEL_FILE = DATA_DIR / "titanic_model.pt"

FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


def preprocess_dataframe(dataframe):
    df = dataframe.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype("float32")
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2}).astype("float32")
    df["Age"] = df["Age"].fillna(df["Age"].median()).astype("float32")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median()).astype("float32")
    df["Pclass"] = df["Pclass"].astype("float32")
    df["SibSp"] = df["SibSp"].astype("float32")
    df["Parch"] = df["Parch"].astype("float32")
    return df


class TitanicDataset(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        df = preprocess_dataframe(df)
        x = df[FEATURE_COLUMNS].values
        y = df[["Survived"]].values

        self.x_data = torch.tensor(x, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.float32)
        self.len = len(df)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(len(FEATURE_COLUMNS), 12)
        self.linear2 = torch.nn.Linear(12, 8)
        self.linear3 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    dataset = TitanicDataset(TRAIN_FILE)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Model()
    criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        epoch_loss = 0.0
        for batch_index, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"epoch={epoch:03d} batch={batch_index:03d} loss={loss.item():.6f}")

        average_loss = epoch_loss / len(train_loader)
        print(f"epoch={epoch:03d} avg_loss={average_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_FILE,
    )
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
