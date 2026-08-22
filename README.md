# LA Crime Analytics — Urban Safety Dashboard

An end-to-end data analytics project on 740K+ LAPD crime incident records (2020–present), built to surface geographic and temporal crime patterns and to predict likely crime type from situational features. Includes a full reproducible pipeline (cleaning → feature engineering → modeling) and an interactive Streamlit dashboard.

**Live dashboard:** [https://la-crime-analytics.streamlit.app/]

---

## Overview

Urban safety analysis matters to city planners, law enforcement resource allocation, and public awareness. This project takes raw, messy incident-level crime data from the City of Los Angeles and turns it into:

- A **cleaned, reproducible dataset** (740K+ records, 24 → 32 columns after feature engineering)
- **Exploratory analysis** of crime patterns by area, time of day, day of week, and season
- A **Random Forest classifier** predicting the most likely crime type given location, time, and victim demographics
- An **interactive dashboard** for filtering, visualizing hotspots on a map, and running live predictions

## Dataset

- **Source:** [LAPD Crime Data from 2020 to Present](https://www.kaggle.com/datasets) (via Kaggle, originally published by the City of Los Angeles at [data.lacity.org](https://data.lacity.org))
- **Size:** ~744,000 incident records, 28 raw columns
- **Fields:** date/time of occurrence, area, crime code and description, victim age/sex/descent, premise type, weapon used, case status, and geolocation (lat/lon)

The raw CSV is not committed to this repository due to size — see [Setup](#setup) to download it.

## Project structure

```
la-crime-analytics/
├── data/
│   ├── raw/                          # place la_crime_data.csv here
│   └── processed/                    # generated: cleaned + feature-engineered CSVs
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_analysis.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── data_cleaning.py              # reusable cleaning pipeline
│   ├── feature_engineering.py        # derives time/violence/area features
│   └── train_model.py                # trains + saves the classifier
├── models/                           # generated: trained model + encoders
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── main.py                           # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Pipeline

**1. Data cleaning** (`src/data_cleaning.py`)
Parses inconsistent date formats, drops sparse columns (`Crm Cd 2-4`, `Cross Street`), removes rows missing critical fields, imputes moderate-missingness fields (victim sex/descent, weapon), filters invalid ages, and deduplicates.
`743,817 → 743,328 rows`

**2. Feature engineering** (`src/feature_engineering.py`)
Derives features used throughout the analysis and model:
| Feature | Description |
|---|---|
| `Crime Hour` | Hour of day (0–23) extracted from `TIME OCC` |
| `Day of Week`, `Month`, `Month Name`, `Year` | Calendar features from `DATE OCC` |
| `Weapon Used Flag` | 1 if a weapon code is present |
| `Violent Crime Flag` | 1 if crime description matches LAPD Part 1 violent crime categories (assault, battery, robbery, rape, homicide, etc.) |
| `Area Crime Count` | Total historical incident volume for that patrol area |

**3. Model training** (`src/train_model.py`)
A Random Forest classifier predicts crime type (narrowed to the 10 most frequent categories for a well-posed multi-class problem) from area, time, victim demographics, and the engineered flags.

- **Accuracy:** 51.1% across 10 balanced classes (vs. ~10% random baseline)
- Strongest performance on **Vehicle — Stolen** (90% F1), weakest on **Theft from Motor Vehicle — Petty** (4% F1), reflecting how distinguishable each crime type is from situational features alone — vehicle theft correlates strongly with time/place, while petty theft from a vehicle looks statistically similar to several other property crimes
- Full precision/recall/F1 breakdown is in `notebooks/04_model_training.ipynb`

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides:

- **Overview** — top crime types, violent/non-violent split, incidents by area
- **Geographic** — density heatmap of incident locations across LA
- **Temporal Patterns** — incidents by hour, day of week, and month
- **Predict Crime Type** — interactive form to get live model predictions for a given area, time, and profile

Filters (year, area, violent/non-violent, hour range) apply across all tabs.

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd la-crime-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Get "Crime Data from 2020 to Present" from Kaggle / data.lacity.org
# and place it at data/raw/la_crime_data.csv

# 4. Run the full pipeline (cleaning -> feature engineering -> model training)
python main.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

The pipeline takes roughly 2–3 minutes on the full dataset (mostly model training).

## Tech stack

- **Data processing:** pandas, numpy
- **Modeling:** scikit-learn (Random Forest)
- **Visualization:** Plotly, matplotlib, seaborn
- **Dashboard:** Streamlit
- **Environment:** Python 3.12, Jupyter

## Key findings

- Violent crime accounts for roughly **29%** of all incidents in this dataset
- Crime volume peaks in the **evening hours**, consistent with typical urban crime patterns
- Incident counts vary substantially by patrol area, with the highest-volume areas seeing more than double the incidents of the lowest
- Vehicle theft is the single most predictable crime type from situational features alone — likely because it correlates strongly with location and time in a way that assault-type crimes do not

## Possible extensions

- Deploy the dashboard publicly via Streamlit Community Cloud
- Add year-over-year trend comparison
- Incorporate demographic/census overlays for socioeconomic context
- Experiment with gradient boosting (XGBoost/LightGBM) or class-balanced sampling to improve minority-class recall
- Add a time-series forecasting component (e.g., predicting next month's incident volume by area)

## Author

Built as a personal project applying end-to-end data analytics — from raw data to a deployed, interactive tool — to a real-world public safety dataset.
