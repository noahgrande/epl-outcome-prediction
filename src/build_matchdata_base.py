"""
BUILD MATCHDATA BASE — ENRICHED VERSION (NO PIVOT)

This script:
1. Loads matchdata_21-25.csv
2. Normalizes ALL team names + referee names
3. Renames columns to clean ML-friendly names
4. Keeps all important features (xG, shooting, passing, defense, creation…)
5. Adds numeric matchweek
6. Sorts by season > matchweek > team > date
7. Saves clean base dataset (no pivot yet)
"""

import pandas as pd
import re

RAW_FILE = "data/raw/matchdata_21-25.csv"
OUT_FILE = "data/processed/matchdata_base.csv"


# ----------------------------------------------------------
# TEAM NORMALIZATION
# ----------------------------------------------------------

TEAM_NORMALIZATION = {
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "manchester united": "manchester united",
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "mu": "manchester united",
    "tottenham": "tottenham hotspur",
    "tottenham hotspurs": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "spurs": "tottenham hotspur",
    "aston villa": "aston villa",
    "villa": "aston villa",
    "bournemouth": "bournemouth",
    "afc bournemouth": "bournemouth",
    "brentford": "brentford",
    "brighton": "brighton and hove albion",
    "brighton hove albion": "brighton and hove albion",
    "brighton and hove albion": "brighton and hove albion",
    "burnley": "burnley",
    "crystal palace": "crystal palace",
    "palace": "crystal palace",
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
    "nott'm forest": "nottingham forest",
    "norwich": "norwich city",
    "norwich city": "norwich city",
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

def normalize_formation(f):
    """
    Standardise any formation coming from fbref weird formats:
    - '3/5/2002' → '3-5-2'
    - '4/4/2002' → '4-4-2'
    - '4/3/2003' → '4-3-3'
    - keeps '4-2-3-1' unchanged
    """
    if pd.isna(f):
        return f

    f = str(f).strip()

    # Already clean
    if "-" in f and "/" not in f:
        return f

    # Detect fbref broken formations: extract all leading digits
    # Example: "3/5/2002" → ["3", "5", "2002"]
    nums = re.findall(r"(\d+)", f)

    # If we don't have at least 2 numbers → can't use
    if len(nums) < 2:
        return f

    # Typical pattern: first 2 numbers = formation, last number = year artifact
    # Example ["3","5","2002"] → 3-5-2
    if len(nums) == 3:
        return f"{nums[0]}-{nums[1]}-{nums[2][0]}"

    # Rare weird cases fallback
    if len(nums) == 2:
        return f"{nums[0]}-{nums[1]}"

    # More than 3 numbers? keep first 3
    return "-".join(nums[:3])



# ----------------------------------------------------------
# REFEREE NORMALIZATION (remove initials, unify spacing)
# ----------------------------------------------------------

def normalize_referee(x):
    if pd.isna(x):
        return x
    x = x.strip()
    x = x.replace(".", "")
    x = x.replace("  ", " ")
    x = x.title()
    return x


# ----------------------------------------------------------
# COLUMN RENAME MAP
# ----------------------------------------------------------

COLUMN_RENAME = {
    "date": "match_date",
    "round": "matchweek",

    "gf": "goals_for",
    "ga": "goals_against",

    "xg": "xg",
    "npxg": "non_penalty_xg",
    "xga": "xg_against",
    "psxg": "post_shot_xg",
    "g-xg": "goals_minus_xg",
    "psxg+/-": "post_shot_xg_diff",

    "sh": "shots",
    "sot": "shots_on_target",
    "dist": "avg_shot_distance",

    "sota": "shots_on_target_against",
    "saves": "saves",
    "save_pct": "save_percentage",
    "cs": "clean_sheets",

    "poss": "possession",

    "fk": "free_kicks",
    "pk": "penalties_scored",
    "pkatt": "penalties_attempted",

    "cmp": "passes_completed",
    "att": "passes_attempted",
    "cmp_pct": "pass_completion_pct",
    "totdist": "total_distance_progressed",
    "prgdist": "progressive_distance",
    "prgc": "progressive_carries",

    "ast": "assists",
    "xag": "expected_assisted_goals",
    "xa": "expected_assists",
    "kp": "key_passes",
    "sca": "shot_creating_actions",
    "gca": "goal_creating_actions",

    "mis": "miscontrols",
    "dis": "dispossessed",
    "rec": "recoveries",

    "tkl": "tackles",
    "tklw": "tackles_won",
    "int": "interceptions",
    "tkl+int": "defensive_actions",
    "blocks": "blocks",
    "clr": "clearances",

    "formation": "team_formation",
    "opp_formation": "opponent_formation",
}


# ----------------------------------------------------------
# COLUMNS TO KEEP (standardized)
# ----------------------------------------------------------

USEFUL_COLS = [
    "team", "season", "match_date", "matchweek", "competition", "venue",
    "opponent", "referee",

    "goals_for", "goals_against",

    "xg", "non_penalty_xg", "xg_against", "post_shot_xg",
    "goals_minus_xg", "post_shot_xg_diff",

    "shots", "shots_on_target", "avg_shot_distance",

    "shots_on_target_against", "saves", "save_percentage", "clean_sheets",

    "possession",

    "free_kicks", "penalties_scored", "penalties_attempted",

    "passes_completed", "passes_attempted", "pass_completion_pct",
    "total_distance_progressed", "progressive_distance",
    "progressive_carries",

    "assists", "expected_assisted_goals", "expected_assists",
    "key_passes", "shot_creating_actions", "goal_creating_actions",

    "miscontrols", "dispossessed", "recoveries",

    "tackles", "tackles_won", "interceptions",
    "defensive_actions", "blocks", "clearances",

    "team_formation", "opponent_formation",
]


# ----------------------------------------------------------
# MAIN LOADING FUNCTION
# ----------------------------------------------------------

def load_raw():
    df = pd.read_csv(RAW_FILE)

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

    # normalize teams
    df["team"] = df["team"].apply(normalize_team)
    df["opponent"] = df["opponent"].apply(normalize_team)

    # normalize referees
    df["referee"] = df["referee"].apply(normalize_referee)

    # rename columns
    df = df.rename(columns=COLUMN_RENAME)

    # NOW normalize formations
    for col in ["team_formation", "opponent_formation"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_formation)

    # matchweek number
    df["matchweek_num"] = df["matchweek"].str.extract(r"(\d+)").astype(int)

    # keep only useful columns
    keep = [c for c in USEFUL_COLS if c in df.columns]
    df = df[keep + ["matchweek_num"]]

    df = df.sort_values(["season", "matchweek_num", "team", "match_date"])

    return df


# ----------------------------------------------------------
# EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":

    print("Loading raw enriched matchdata…")
    df = load_raw()

    print("Final shape:", df.shape)
    df.to_csv(OUT_FILE, index=False)

    print(f"\n✔ Saved CLEAN ENRICHED BASE dataset → {OUT_FILE}")

