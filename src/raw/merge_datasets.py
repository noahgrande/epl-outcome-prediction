import pandas as pd

MATCHDATA_FILE = "data/processed/matchdata_clean.csv"
ALLMATCHES_FILE = "data/processed/all_matches_clean.csv"
OUT_FILE = "data/processed/data_merged.csv"

print("Loading matchdata_clean …")
matchdata = pd.read_csv(MATCHDATA_FILE)

print("Loading all_matches_clean …")
allm = pd.read_csv(ALLMATCHES_FILE)

# ---------------------------------------------------------
# Colonnes de all_matches_clean qui doublonnent matchdata_clean
# (on garde la version matchdata_clean)
# ---------------------------------------------------------
DUPLICATE_INTENT_COLS = [
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "referee",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
]

# On enlève ces colonnes uniquement du fichier all_matches
cols_to_keep_allm = [c for c in allm.columns if c not in DUPLICATE_INTENT_COLS]
allm = allm[cols_to_keep_allm]

print("Columns in matchdata_clean:", matchdata.columns.tolist())
print("Columns kept from all_matches_clean:", allm.columns.tolist())

# ---------------------------------------------------------
# MERGE sur match_id
# ---------------------------------------------------------
print("\nMerging on 'match_id' …")
merged = matchdata.merge(allm, on="match_id", how="left")

print("Final shape:", merged.shape)

merged.to_csv(OUT_FILE, index=False)
print(f"\n✔ Saved merged dataset → {OUT_FILE}")





