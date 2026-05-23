import json
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split


DATA_DIR = Path(__file__).resolve().parent / "titanic"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = DATA_DIR / "titanic_model.pt"
METADATA_FILE = DATA_DIR / "titanic_features.json"

TITLE_GROUPS = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Sir": "Royalty",
    "Don": "Royalty",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Mlle": "Miss",
}

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FamilySize",
    "FarePerPerson",
    "TicketGroupSize",
    "CabinCount",
]


def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", name)
    if not match:
        return "Rare"
    title = match.group(1).strip()
    return TITLE_GROUPS.get(title, "Rare")


def extract_ticket_prefix(ticket):
    cleaned = str(ticket).replace(".", "").replace("/", "").strip()
    parts = cleaned.split()
    if len(parts) <= 1:
        return "NONE"
    return parts[0]


def extract_deck(cabin):
    if pd.isna(cabin):
        return "U"
    return str(cabin)[0]


def build_features(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    train["dataset"] = "train"
    test["dataset"] = "test"
    test["Survived"] = -1

    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined["Title"] = combined["Name"].apply(extract_title)
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype("int64")
    combined["TicketPrefix"] = combined["Ticket"].apply(extract_ticket_prefix)
    combined["TicketGroupSize"] = combined.groupby("Ticket")["Ticket"].transform("count")
    combined["Deck"] = combined["Cabin"].apply(extract_deck)
    combined["HasCabin"] = combined["Cabin"].notna().astype("int64")
    combined["CabinCount"] = combined["Cabin"].fillna("").apply(lambda x: len(str(x).split()) if x else 0)

    embarked_mode = combined["Embarked"].dropna().mode().iloc[0]
    combined["Embarked"] = combined["Embarked"].fillna(embarked_mode)

    fare_by_pclass = combined.groupby("Pclass")["Fare"].median()
    combined["Fare"] = combined.apply(
        lambda row: fare_by_pclass.loc[row["Pclass"]] if pd.isna(row["Fare"]) else row["Fare"],
        axis=1,
    )

    age_by_title_pclass = combined.groupby(["Title", "Pclass"])["Age"].median()
    global_age_median = combined["Age"].median()

    def fill_age(row):
        if not pd.isna(row["Age"]):
            return row["Age"]
        key = (row["Title"], row["Pclass"])
        age_value = age_by_title_pclass.get(key, float("nan"))
        if pd.isna(age_value):
            return global_age_median
        return age_value

    combined["Age"] = combined.apply(fill_age, axis=1)
    combined["FarePerPerson"] = combined["Fare"] / combined["FamilySize"]

    combined["Sex"] = combined["Sex"].map({"male": "male", "female": "female"})

    model_frame = combined[
        [
            "dataset",
            "PassengerId",
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Title",
            "FamilySize",
            "IsAlone",
            "TicketPrefix",
            "TicketGroupSize",
            "Deck",
            "HasCabin",
            "CabinCount",
            "FarePerPerson",
        ]
    ].copy()

    categorical_columns = ["Pclass", "Sex", "Embarked", "Title", "TicketPrefix", "Deck"]
    model_frame[categorical_columns] = model_frame[categorical_columns].astype(str)
    model_frame = pd.get_dummies(model_frame, columns=categorical_columns, dtype="float32")

    feature_columns = [
        column
        for column in model_frame.columns
        if column not in {"dataset", "PassengerId", "Survived"}
    ]

    train_features = model_frame[model_frame["dataset"] == "train"].copy()
    test_features = model_frame[model_frame["dataset"] == "test"].copy()

    scaling = {}
    for column in NUMERIC_FEATURES:
        mean_value = float(train_features[column].mean())
        std_value = float(train_features[column].std())
        if std_value == 0.0:
            std_value = 1.0
        scaling[column] = {"mean": mean_value, "std": std_value}
        train_features[column] = (train_features[column] - mean_value) / std_value
        test_features[column] = (test_features[column] - mean_value) / std_value

    metadata = {
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "scaling": scaling,
    }
    return train_features, test_features, metadata


class TitanicDataset(Dataset):
    def __init__(self, dataframe, feature_columns):
        self.x_data = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32)
        self.y_data = torch.tensor(dataframe[["Survived"]].values, dtype=torch.float32)
        self.len = len(dataframe)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    with torch.no_grad():
        for inputs, labels in data_loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return average_loss, accuracy


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    train_features, _, metadata = build_features(train_df, test_df)
    feature_columns = metadata["feature_columns"]

    print(f"train rows={len(train_features)}")
    print(f"feature count={len(feature_columns)}")
    print(f"sample features={feature_columns[:12]}")

    dataset = TitanicDataset(train_features, feature_columns)
    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, valid_subset = random_split(dataset, [train_size, valid_size], generator=generator)

    train_loader = DataLoader(dataset=train_subset, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_subset, batch_size=64, shuffle=False, num_workers=0)

    model = Model(len(feature_columns))
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_valid_accuracy = 0.0
    best_state_dict = None

    for epoch in range(120):
        model.train()
        total_loss = 0.0

        for batch_index, data in enumerate(train_loader, 0):
            inputs, labels = data
            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_index % 5 == 0:
                print(f"epoch={epoch:03d} batch={batch_index:03d} loss={loss.item():.6f}")

        train_loss = total_loss / len(train_loader)
        valid_loss, valid_accuracy = evaluate(model, valid_loader)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"valid_loss={valid_loss:.6f} valid_acc={valid_accuracy:.4f}"
        )

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state_dict is None:
        best_state_dict = model.state_dict()

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "feature_columns": feature_columns,
        },
        MODEL_FILE,
    )
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")
    print(f"best_valid_accuracy={best_valid_accuracy:.4f}")


if __name__ == "__main__":
    main()
