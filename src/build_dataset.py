"""
Script to build a clean EPL dataset from multiple raw CSV files.

Steps:
1. Load all season files (19_20, 20_21, ..., 23_24)
2. Keep only useful columns
3. Clean column names
4. Concatenate into a single DataFrame
5. Save final dataset in data/processed
"""

import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# 1. Load and clean a single season file
# ---------------------------------------------------------
def load_season_file(path: str) -> pd.DataFrame:
    """Load a single raw CSV file and keep only useful columns."""
    
    # Useful columns from football-data.uk (your 23_24 example)
    USEFUL_COLS = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "Referee",
        "HS", "AS", "HST", "AST",
        "B365H", "B365D", "B365A",
        "PSH", "PSD", "PSA"
    ]

    df = pd.read_csv(path)

    # Keep only columns that exist in the file
    df = df[[col for col in USEFUL_COLS if col in df.columns]]

    # Standardize column names (lowercase, underscores)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )

    return df


# ---------------------------------------------------------
# 2. Load all EPL season files from /data/raw
# ---------------------------------------------------------
def load_all_seasons(raw_path: str) -> pd.DataFrame:
    """Load multiple CSV files from data/raw and combine them."""
    
    path = Path(raw_path)
    csv_files = sorted(list(path.glob("*.csv")))

    print(f"Found {len(csv_files)} raw season files.")

    dfs = []

    for f in csv_files:
        print(f"Loading {f.name}...")
        df = load_season_file(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ---------------------------------------------------------
# 3. Save processed dataset
# ---------------------------------------------------------
def save_dataset(df: pd.DataFrame, out_path: str):
    """Save final dataset in data/processed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset → {out_path}")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    RAW_DIR = "../data/raw"        # relative path from src/
    OUT_FILE = "../data/processed/all_matches_clean.csv"

    df = load_all_seasons(RAW_DIR)
    save_dataset(df, OUT_FILE)

    print("Dataset build complete.")
