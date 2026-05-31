import json
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).resolve().parent / "titanic"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = DATA_DIR / "titanic_model.pt"
METADATA_FILE = DATA_DIR / "titanic_features.json"

RANDOM_SEED = 42
VALID_RATIO = 0.2
EPOCHS = 120
BATCH_SIZE = 32

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
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "FarePerPerson",
    "TicketGroupSize",
    "CabinCount",
]
BINARY_FEATURES = ["IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "TicketPrefix", "Deck"]


def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", str(name))
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


def split_train_valid(dataframe):
    train_size = int(len(dataframe) * (1 - VALID_RATIO))
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(len(dataframe), generator=generator).tolist()
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    return (
        dataframe.iloc[train_indices].reset_index(drop=True),
        dataframe.iloc[valid_indices].reset_index(drop=True),
    )


def add_derived_features(dataframe, ticket_group_sizes=None):
    df = dataframe.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype("int64")
    df["TicketPrefix"] = df["Ticket"].apply(extract_ticket_prefix)
    df["Deck"] = df["Cabin"].apply(extract_deck)
    df["HasCabin"] = df["Cabin"].notna().astype("int64")
    df["CabinCount"] = df["Cabin"].fillna("").apply(lambda x: len(str(x).split()) if x else 0)

    if ticket_group_sizes is None:
        df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    else:
        df["TicketGroupSize"] = df["Ticket"].astype(str).map(ticket_group_sizes).fillna(1)

    return df


def first_mode_or_default(series, default_value):
    mode = series.dropna().mode()
    if mode.empty:
        return default_value
    return mode.iloc[0]


def fit_preprocessor(dataframe):
    df = add_derived_features(dataframe)

    embarked_mode = first_mode_or_default(df["Embarked"], "S")
    fare_median = float(df["Fare"].median())
    if pd.isna(fare_median):
        fare_median = 0.0

    fare_by_pclass = {
        str(pclass): float(fare)
        for pclass, fare in df.groupby("Pclass")["Fare"].median().dropna().items()
    }

    age_median = float(df["Age"].median())
    if pd.isna(age_median):
        age_median = 30.0

    age_by_title_pclass = {
        f"{title}|{pclass}": float(age)
        for (title, pclass), age in df.groupby(["Title", "Pclass"])["Age"].median().dropna().items()
    }

    ticket_group_sizes = {
        str(ticket): int(count)
        for ticket, count in df["Ticket"].astype(str).value_counts().items()
    }

    df = apply_fill_values(
        df,
        {
            "embarked_mode": embarked_mode,
            "fare_median": fare_median,
            "fare_by_pclass": fare_by_pclass,
            "age_median": age_median,
            "age_by_title_pclass": age_by_title_pclass,
        },
    )

    category_values = {
        column: sorted(df[column].astype(str).dropna().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }

    scaling = {}
    for column in NUMERIC_FEATURES:
        mean_value = float(df[column].mean())
        std_value = float(df[column].std())
        if pd.isna(std_value) or std_value == 0.0:
            std_value = 1.0
        scaling[column] = {"mean": mean_value, "std": std_value}

    feature_columns = [*NUMERIC_FEATURES, *BINARY_FEATURES]
    for column in CATEGORICAL_FEATURES:
        feature_columns.extend([f"{column}_{value}" for value in category_values[column]])

    return {
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_values": category_values,
        "fill_values": {
            "embarked_mode": embarked_mode,
            "fare_median": fare_median,
            "fare_by_pclass": fare_by_pclass,
            "age_median": age_median,
            "age_by_title_pclass": age_by_title_pclass,
        },
        "scaling": scaling,
        "ticket_group_sizes": ticket_group_sizes,
    }


def apply_fill_values(dataframe, fill_values):
    df = dataframe.copy()
    df["Embarked"] = df["Embarked"].fillna(fill_values["embarked_mode"])

    def fill_fare(row):
        if not pd.isna(row["Fare"]):
            return row["Fare"]
        return fill_values["fare_by_pclass"].get(str(row["Pclass"]), fill_values["fare_median"])

    def fill_age(row):
        if not pd.isna(row["Age"]):
            return row["Age"]
        key = f"{row['Title']}|{row['Pclass']}"
        return fill_values["age_by_title_pclass"].get(key, fill_values["age_median"])

    df["Fare"] = df.apply(fill_fare, axis=1)
    df["Age"] = df.apply(fill_age, axis=1)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["Sex"] = df["Sex"].fillna("unknown")
    return df


def transform_features(dataframe, metadata):
    df = add_derived_features(dataframe, metadata["ticket_group_sizes"])
    df = apply_fill_values(df, metadata["fill_values"])

    features = pd.DataFrame(index=df.index)
    if "PassengerId" in df.columns:
        features["PassengerId"] = df["PassengerId"]
    if "Survived" in df.columns:
        features["Survived"] = df["Survived"]

    for column in NUMERIC_FEATURES:
        mean_value = metadata["scaling"][column]["mean"]
        std_value = metadata["scaling"][column]["std"]
        features[column] = ((df[column] - mean_value) / std_value).astype("float32")

    for column in BINARY_FEATURES:
        features[column] = df[column].astype("float32")

    for column in CATEGORICAL_FEATURES:
        values = df[column].astype(str)
        for category in metadata["category_values"][column]:
            features[f"{column}_{category}"] = (values == category).astype("float32")

    return features.reindex(columns=metadata["feature_columns"] + metadata_columns(features), fill_value=0.0)


def metadata_columns(dataframe):
    return [column for column in ["PassengerId", "Survived"] if column in dataframe.columns]


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


def train_model(train_features, feature_columns, valid_features=None, phase_name="train"):
    train_dataset = TitanicDataset(train_features, feature_columns)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = None
    if valid_features is not None:
        valid_dataset = TitanicDataset(valid_features, feature_columns)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = Model(len(feature_columns))
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_valid_accuracy = 0.0
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        if valid_loader is None:
            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
                print(f"{phase_name} epoch={epoch:03d} train_loss={train_loss:.6f}")
            continue

        valid_loss, valid_accuracy = evaluate(model, valid_loader)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            print(
                f"{phase_name} epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"valid_loss={valid_loss:.6f} valid_acc={valid_accuracy:.4f}"
            )

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_valid_accuracy


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    raw_train_df = pd.read_csv(TRAIN_FILE)
    train_raw_df, valid_raw_df = split_train_valid(raw_train_df)

    validation_metadata = fit_preprocessor(train_raw_df)
    validation_feature_columns = validation_metadata["feature_columns"]
    train_features = transform_features(train_raw_df, validation_metadata)
    valid_features = transform_features(valid_raw_df, validation_metadata)

    print(f"train rows={len(train_features)} valid rows={len(valid_features)}")
    print(f"validation feature count={len(validation_feature_columns)}")
    _, best_valid_accuracy = train_model(
        train_features,
        validation_feature_columns,
        valid_features=valid_features,
        phase_name="validation",
    )

    final_metadata = fit_preprocessor(raw_train_df)
    final_feature_columns = final_metadata["feature_columns"]
    final_train_features = transform_features(raw_train_df, final_metadata)
    final_model, _ = train_model(
        final_train_features,
        final_feature_columns,
        valid_features=None,
        phase_name="final",
    )

    final_metadata["validation"] = {
        "seed": RANDOM_SEED,
        "valid_ratio": VALID_RATIO,
        "best_valid_accuracy": best_valid_accuracy,
        "train_rows": len(train_raw_df),
        "valid_rows": len(valid_raw_df),
    }

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "feature_columns": final_feature_columns,
        },
        MODEL_FILE,
    )
    METADATA_FILE.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")
    print(f"final feature count={len(final_feature_columns)}")
    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")
    print(f"best_valid_accuracy={best_valid_accuracy:.4f}")


if __name__ == "__main__":
    main()
