"""
Data Cleaning Module
=====================
Loads the raw LA Crime dataset, cleans it, and saves a processed version.

Run directly:
    python src/data_cleaning.py
"""

import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "la_crime_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "cleaned_crime_data.csv")


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw crime dataset from disk."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps and return a tidy DataFrame."""

    # Convert date columns
    df["Date Rptd"] = pd.to_datetime(
        df["Date Rptd"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    df["DATE OCC"] = pd.to_datetime(
        df["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )

    # Drop sparse / mostly-empty columns
    columns_to_drop = ["Crm Cd 2", "Crm Cd 3", "Crm Cd 4", "Cross Street"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Drop rows missing critical fields
    df = df.dropna(subset=["Premis Cd", "Premis Desc", "Crm Cd 1"]).copy()

    # Fill moderate null values
    df["Vict Sex"] = df["Vict Sex"].fillna("Unknown")
    df["Vict Descent"] = df["Vict Descent"].fillna("Unknown")
    df["Mocodes"] = df["Mocodes"].fillna("Not Specified")
    df["Weapon Used Cd"] = df["Weapon Used Cd"].fillna(0)
    df["Weapon Desc"] = df["Weapon Desc"].fillna("No Weapon")

    # Clean victim age (drop invalid / placeholder ages)
    df = df[(df["Vict Age"] >= 0) & (df["Vict Age"] <= 100)].copy()

    # Drop duplicate rows
    df = df.drop_duplicates().copy()

    return df


def main():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")

    print("Cleaning data...")
    df_clean = clean_data(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Remaining nulls:\n{df_clean.isnull().sum()}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
