"""
Build a clean EPL dataset from football-data.co.uk raw season files.

Covers seasons 2021–2022 → 2024–2025.
Fixes date formats, standardizes columns, and concatenates seasons.
"""

import pandas as pd
from pathlib import Path


# ---------------------------------------------
# Columns we keep from football-data.co.uk
# ---------------------------------------------
USEFUL_COLS = [
    "Date",
    "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "Referee",
    "HS", "AS", "HST", "AST",
    "B365H", "B365D", "B365A",
    "PSH", "PSD", "PSA",
]


# ---------------------------------------------------------
# 1. Load and clean one season file
# ---------------------------------------------------------
def load_season_file(path: Path) -> pd.DataFrame:
    print(f" → Loading {path.name}")

    df = pd.read_csv(path)

    # Keep only useful columns that exist
    cols = [c for c in USEFUL_COLS if c in df.columns]
    df = df[cols].copy()

    # Fix date parsing (important!)
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="coerce",
        dayfirst=True
    )

    # Remove invalid dates
    df = df.dropna(subset=["Date"])

    # Force final format YYYY-MM-DD
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Standardize names
    df.columns = (
        df.columns.str.lower()
                  .str.strip()
                  .str.replace(" ", "_")
    )

    # Add source file column for debugging
    df["source_file"] = path.name

    return df


# ---------------------------------------------------------
# 2. Load all seasons and concatenate
# ---------------------------------------------------------
def load_all_seasons(raw_folder: str) -> pd.DataFrame:
    raw_path = Path(raw_folder)

    # Only seasons 21_22 to 24_25
    files = sorted([
        f for f in raw_path.glob("*.csv")
        if f.name.startswith(("21_22", "22_23", "23_24", "24_25"))
    ])

    print(f"\nLoading EPL seasons 2021–2025...")
    print(f"Detected {len(files)} files:")
    for f in files:
        print("  -", f.name)

    dfs = []
    for f in files:
        dfs.append(load_season_file(f))

    combined = pd.concat(dfs, ignore_index=True)

    return combined


# ---------------------------------------------------------
# 3. Save clean dataset
# ---------------------------------------------------------
def save_dataset(df: pd.DataFrame, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\n✔ Saved clean EPL dataset →", out_path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    RAW_DIR = "data/raw"
    OUT_FILE = "data/processed/all_matches_clean.csv"

    df = load_all_seasons(RAW_DIR)

    print("\nFinal dataset shape:", df.shape)

    save_dataset(df, OUT_FILE)

    print("\n✔ Dataset build complete.")

