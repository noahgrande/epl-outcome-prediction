"""
BUILD MATCHDATA BASE — STEP 1 (NO PIVOT)

This script:
1. Loads matchdata_21-25.csv
2. Normalizes ALL team names into a single canonical name
3. Renames columns to clear human-readable names
4. Keeps only the useful columns
5. Adds numeric matchweek
6. Sorts by season > matchweek > team > date
7. Saves a clean, ordered base dataset
"""

import pandas as pd

RAW_FILE = "data/raw/matchdata_21-25.csv"
OUT_FILE = "data/processed/matchdata_base.csv"


# ----------------------------------------------------------
# NORMALISATION DES NOMS D'ÉQUIPES
# ----------------------------------------------------------

TEAM_NORMALIZATION = {
    # Big 6
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "manchester united": "manchester united",
    "man united": "manchester united",
    "man utd": "manchester united",
    "mu" : "manchester united",
    "manchester utd": "manchester united",

    # Tottenham
    "tottenham": "tottenham hotspur",
    "tottenham hotspurs": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "spurs": "tottenham hotspur",

    # Other EPL teams
    "aston villa": "aston villa",
    "villa": "aston villa",

    "bournemouth": "bournemouth",
    "afc bournemouth": "bournemouth",

    "brentford": "brentford",

    "brighton": "brighton and hove albion",
    "brighton and hove albion": "brighton and hove albion",
    "brighton hove albion": "brighton and hove albion",

    "burnley": "burnley",

    "crystal palace": "crystal palace",

    "everton": "everton",

    "fulham": "fulham",

    "ipswich": "ipswich town",
    "ipswich town": "ipswich town",

    "leeds": "leeds united",
    "leeds united": "leeds united",

    "leicester": "leicester city",
    "leicester city": "leicester city",

    "luton": "luton town",
    "luton town": "luton town",

    "newcastle": "newcastle united",
    "newcastle utd": "newcastle united",
    "newcastle united": "newcastle united",

    "forest": "nottingham forest",
    "nottingham": "nottingham forest",
    "nottingham forest": "nottingham forest",
    "nott'ham forest": "nottingham forest",

    "sheffield": "sheffield united",
    "sheffield utd": "sheffield united",
    "sheffield united": "sheffield united",

    "southampton": "southampton",

    "west ham": "west ham united",
    "west ham utd": "west ham united",
    "west ham united": "west ham united",

    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "wolverhampton wanderers": "wolverhampton wanderers",
}


def normalize_team(x):
    x = str(x).lower().strip()
    return TEAM_NORMALIZATION.get(x, x)


# ----------------------------------------------------------
# COLUMN RENAMING (MAKE NAMES CLEARER)
# ----------------------------------------------------------

COLUMN_RENAME = {
    "gf": "goals_for",
    "ga": "goals_against",
    "xg": "xg",
    "npxg": "non_penalty_xg",
    "xga": "xg_against",
    "psxg": "post_shot_xg",
    "sh": "shots",
    "sot": "shots_on_target",
    "dist": "avg_shot_distance",
    "sota": "shots_on_target_against",
    "saves": "saves",
    "save_pct": "save_percentage",
    "poss": "possession",
    "fk": "free_kicks",
    "pk": "penalties_scored",
    "pkatt": "penalties_attempted",
    "formation": "team_formation",
    "opp_formation": "opponent_formation",
    "referee": "referee",
}


# ----------------------------------------------------------
# COLUMNS TO KEEP
# ----------------------------------------------------------

USEFUL_COLS = [
    "season", "date", "round", "venue", "comp",
    "team", "opponent",

    # metrics
    "goals_for", "goals_against",
    "xg", "non_penalty_xg", "xg_against", "post_shot_xg",
    "shots", "shots_on_target", "avg_shot_distance",
    "shots_on_target_against", "saves", "save_percentage",
    "possession",
    "free_kicks", "penalties_scored", "penalties_attempted",

    # tactical
    "team_formation", "opponent_formation",

    # referee
    "referee"
]


# ----------------------------------------------------------
# MAIN PROCESS
# ----------------------------------------------------------

def load_raw():
    df = pd.read_csv(RAW_FILE)

    # clean names
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace("%", "pct")
        .str.replace(" ", "_")
        .str.replace(".", "_")
    )

    # parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    bad = df["date"].isna()
    if bad.any():
        df.loc[bad, "date"] = pd.to_datetime(df.loc[bad, "date"], errors="coerce", dayfirst=True)

    # normalize team names
    df["team"] = df["team"].apply(normalize_team)
    df["opponent"] = df["opponent"].apply(normalize_team)

    # rename columns to readable names
    df = df.rename(columns=COLUMN_RENAME)

    # convert matchweek to number
    df["matchweek_num"] = df["round"].str.extract(r"(\d+)").astype(int)

    # keep only relevant columns
    keep = [c for c in USEFUL_COLS if c in df.columns]
    df = df[keep + ["matchweek_num"]]

    # sort cleanly
    df = df.sort_values(["season", "matchweek_num", "team", "date"])

    return df


# ----------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":
    print("Loading raw matchdata…")

    df = load_raw()

    print("Final shape:", df.shape)
    df.to_csv(OUT_FILE, index=False)

    print(f"\n✔ Saved CLEAN BASE dataset → {OUT_FILE}")

