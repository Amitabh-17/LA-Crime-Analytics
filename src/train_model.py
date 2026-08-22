"""
Model Training Module
=======================
Trains a Random Forest classifier to predict crime type from
area, time, and demographic features. Saves the model + encoders.

Run directly:
    python src/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

INPUT_PATH = os.path.join("data", "processed", "feature_engineered_crime_data.csv")
MODEL_DIR = "models"

FEATURES = [
    "AREA NAME",
    "Crime Hour",
    "Day of Week",
    "Month",
    "Vict Age",
    "Vict Sex",
    "Weapon Used Flag",
    "Violent Crime Flag",
    "Area Crime Count",
]
TARGET = "Crm Cd Desc"
TOP_N_CLASSES = 10


def load_data(path: str = INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    # Reduce target to the top N most common crime types (multi-class ML
    # on 100+ rare classes performs poorly and is hard to evaluate)
    top_crimes = df[TARGET].value_counts().head(TOP_N_CLASSES).index
    df = df[df[TARGET].isin(top_crimes)].copy()

    X = df[FEATURES].copy()
    y = df[TARGET]

    encoders = {}
    for col in ["AREA NAME", "Day of Week", "Vict Sex"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, acc, report


def main():
    print("Loading feature-engineered data...")
    df = load_data()

    print("Preparing features...")
    X, y, encoders = prepare_data(df)
    print(f"Training on {len(X)} rows, {y.nunique()} crime classes")

    print("Training Random Forest...")
    model, acc, report = train(X, y)

    print(f"\nAccuracy: {acc:.4f}\n")
    print(report)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "crime_type_model.pkl"), compress=3)
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_list.pkl"))
    print(f"\nModel saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
