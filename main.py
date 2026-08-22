"""
LA Crime Analytics — Pipeline Runner
========================================
Runs the full data pipeline end-to-end: cleaning -> feature engineering
-> model training. Run this once before launching the dashboard.

Usage:
    python main.py              # run full pipeline
    python main.py --skip-model # skip model training (faster)

To launch the dashboard afterward:
    streamlit run dashboard/app.py
"""

import argparse
import subprocess
import sys
import time


def run_step(name: str, script: str):
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print("=" * 60)
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n'{name}' failed after {elapsed:.1f}s. Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"\n'{name}' completed in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Run the LA Crime Analytics pipeline.")
    parser.add_argument("--skip-model", action="store_true", help="Skip model training step")
    args = parser.parse_args()

    print("LA Crime Analytics — Pipeline")
    print("Make sure data/raw/la_crime_data.csv exists before running.\n")

    run_step("Data Cleaning", "src/data_cleaning.py")
    run_step("Feature Engineering", "src/feature_engineering.py")

    if not args.skip_model:
        run_step("Model Training", "src/train_model.py")
    else:
        print("\nSkipping model training (--skip-model).")

    print("\nPipeline complete. Launch the dashboard with:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
