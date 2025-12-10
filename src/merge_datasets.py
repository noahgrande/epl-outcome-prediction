import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
matches = pd.read_csv("data/processed/all_matches_clean.csv")
fbref = pd.read_csv("data/processed/matchdata_clean.csv")

print("Football-data matches:", matches.shape)
print("FBref matches:", fbref.shape)


# ---------------------------------------------------------
# Standardize team names (VERY IMPORTANT)
# ---------------------------------------------------------
def clean_team_name(x):
    return (
        str(x)
        .lower()
        .replace(".", "")
        .replace("  ", " ")
        .strip()
    )

for col in ["home_team", "away_team"]:
    matches[col] = matches[col].apply(clean_team_name)
    fbref[col]   = fbref[col].apply(clean_team_name)


# ---------------------------------------------------------
# Standardize dates
# ---------------------------------------------------------
matches["date"] = pd.to_datetime(matches["date"])
fbref["match_date"] = pd.to_datetime(fbref["match_date"])

matches["merge_date"] = matches["date"].dt.strftime("%Y-%m-%d")
fbref["merge_date"] = fbref["match_date"].dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------
# Merge on date + home + away
# ---------------------------------------------------------
merged = matches.merge(
    fbref,
    left_on=["merge_date", "home_team", "away_team"],
    right_on=["merge_date", "home_team", "away_team"],
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset shape:", merged.shape)


# ---------------------------------------------------------
# Drop helper columns
# ---------------------------------------------------------
merged = merged.drop(columns=["merge_date"])


# ---------------------------------------------------------
# Save final dataset
# ---------------------------------------------------------
OUTPUT = "data/processed/final_dataset_2021_2025.csv"
merged.to_csv(OUTPUT, index=False)
print("\n✔ Saved merged dataset →", OUTPUT)
