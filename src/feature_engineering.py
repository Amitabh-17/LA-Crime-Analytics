"""
Feature Engineering Module
============================
Takes the cleaned crime dataset and derives features used for EDA
and model training: time-based features, violent-crime flag,
weapon-used flag, and area-level crime volume.

Run directly:
    python src/feature_engineering.py
"""

import os
import pandas as pd

INPUT_PATH = os.path.join("data", "processed", "cleaned_crime_data.csv")
OUTPUT_PATH = os.path.join("data", "processed", "feature_engineered_crime_data.csv")

# Keywords used to flag violent offenses from the crime description.
# Based on LAPD's Part 1 violent crime categories: homicide, rape,
# robbery, aggravated assault, and related offenses.
VIOLENT_KEYWORDS = [
    "HOMICIDE", "MANSLAUGHTER", "RAPE", "ROBBERY", "ASSAULT",
    "BATTERY", "KIDNAPPING", "SHOTS FIRED", "CRIMINAL THREATS",
    "LYNCHING", "STALKING",
]


def load_cleaned_data(path: str = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date Rptd", "DATE OCC"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Time-based features ---
    df["Crime Hour"] = (df["TIME OCC"] // 100).astype(int).clip(0, 23)
    df["Day of Week"] = df["DATE OCC"].dt.day_name()
    df["Month"] = df["DATE OCC"].dt.month
    df["Month Name"] = df["DATE OCC"].dt.month_name()
    df["Year"] = df["DATE OCC"].dt.year

    # --- Weapon used flag ---
    df["Weapon Used Flag"] = (df["Weapon Used Cd"] != 0).astype(int)

    # --- Violent crime flag (keyword match on crime description) ---
    pattern = "|".join(VIOLENT_KEYWORDS)
    df["Violent Crime Flag"] = (
        df["Crm Cd Desc"].str.contains(pattern, case=False, na=False).astype(int)
    )

    # --- Area-level crime volume (how many total incidents in that area) ---
    area_counts = df["AREA NAME"].value_counts()
    df["Area Crime Count"] = df["AREA NAME"].map(area_counts)

    return df


def main():
    print("Loading cleaned data...")
    df = load_cleaned_data()
    print(f"Input shape: {df.shape}")

    print("Engineering features...")
    df_fe = engineer_features(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_fe.to_csv(OUTPUT_PATH, index=False)

    print(f"Output shape: {df_fe.shape}")
    print(f"New columns: Crime Hour, Day of Week, Month, Month Name, Year, "
          f"Weapon Used Flag, Violent Crime Flag, Area Crime Count")
    print(f"Violent crime share: {df_fe['Violent Crime Flag'].mean():.2%}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
