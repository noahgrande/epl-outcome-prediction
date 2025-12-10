"""
FINAL MATCHDATA BUILDER — CLEAN + PIVOT (2021–2025)

This script:
1. Loads matchdata_21-25.csv
2. Normalizes column names
3. Normalizes team names (26 official EPL teams)
4. Filters to useful columns
5. Sorts by season > matchweek > team > date
6. Builds a clean match_id
7. Splits dataset into HOME / AWAY
8. Pivots so each match is ONE ROW
9. Writes final clean dataset
"""

import pandas as pd
from pathlib import Path

RAW_FILE = "data/raw/matchdata_21-25.csv"
OUT_PIVOT = "data/processed/matchdata_clean.csv"
OUT_BASE = "data/processed/matchdata_base.csv"


# ============================================================
# TEAM NORMALIZATION TABLE
# ============================================================
TEAM_NORMALIZATION = {
    "arsenal": "arsenal",
    "aston villa": "aston villa",
    "bournemouth": "bournemouth",
    "afc bournemouth": "bournemouth",
    "brentford": "brentford",
    "brighton": "brighton and hove albion",
    "brighton and hove albion": "brighton and hove albion",
    "burnley": "burnley",
    "chelsea": "chelsea",
    "crystal palace": "crystal palace",
    "everton": "everton",
    "fulham": "fulham",
    "ipswich": "ipswich town",
    "ipswich town": "ipswich town",
    "leeds": "leeds united",
    "leeds united": "leeds united",
    "leicester": "leicester city",
    "leicester city": "leicester city",
    "liverpool": "liverpool",
    "luton town": "luton town",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "manchester united": "manchester united",
    "manchester utd": "manchester united",
    "man untd": "manchester united",
    "newcastle utd": "newcastle united",
    "newcastle united": "newcastle united",
    "nottingham forest": "nottingham forest",
    "forest": "nottingham forest",
    "nott'ham forest": "nottingham forest",
    "sheffield": "sheffield united",
    "sheffield utd": "sheffield united",
    "sheffield united": "sheffield united",
    "southampton": "southampton",
    "tottenham": "tottenham hotspur",
    "spurs": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "watford": "watford",
    "west ham": "west ham united",
    "west ham united": "west ham united",
    "wolverhampton wanderers": "wolverhampton wanderers",
    "wolves": "wolverhampton wanderers",
}

def norm(x):
    if pd.isna(x): return x
    x = str(x).lower().strip()
    return TEAM_NORMALIZATION.get(x, x)


# ============================================================
# 1. LOAD + CLEAN BASE
# ============================================================
def load_base():
    df = pd.read_csv(RAW_FILE)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("%", "pct")
        .str.replace(" ", "_")
        .str.replace(".", "_")
    )

    # Parse dates robustly
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    bad = df["date"].isna()
    if bad.any():
        df.loc[bad, "date"] = pd.to_datetime(df.loc[bad, "date"], errors="coerce", dayfirst=True)

    # Normalize names
    df["team"] = df["team"].apply(norm)
    df["opponent"] = df["opponent"].apply(norm)

    return df


# ============================================================
# 2. KEEP ONLY USEFUL COLUMNS
# ============================================================
USEFUL = [
    "season", "team", "opponent", "date", "round", "venue", "comp",
    "gf", "ga",
    "xg", "npxg", "xga", "psxg",
    "sh", "sot", "dist",
    "sota", "saves", "save_pct",
    "poss", "fk", "pk", "pkatt",
    "formation", "opp_formation",
    "referee",
]

def keep_useful(df):
    cols = [c for c in USEFUL if c in df.columns]
    return df[cols].copy()


# ============================================================
# 3. SORTING
# ============================================================
def sort_data(df):
    df["matchweek_num"] = df["round"].str.extract(r"(\d+)").astype(int)
    return df.sort_values(["season", "matchweek_num", "team", "date"])


# ============================================================
# 4. BUILD MATCH ID + SPLIT HOME/AWAY
# ============================================================
def build_match_id(df):
    df["is_home"] = df["venue"].str.lower().eq("home")

    df["match_id"] = (
        df["date"].astype(str)
        + "_"
        + df[["team", "opponent"]].min(axis=1)
        + "_"
        + df[["team", "opponent"]].max(axis=1)
    )
    return df


def split_home_away(df):
    return df[df["is_home"]].copy(), df[~df["is_home"]].copy()


# ============================================================
# 5. PIVOT → ONE MATCH PER ROW
# ============================================================
def pivot(home, away):
    merged = home.merge(
        away,
        on=["match_id", "date"],
        suffixes=("_home", "_away"),
        validate="one_to_one"
    )

    out = pd.DataFrame()

    # Helper: safely read columns
    def safe(col_home, col_away):
        return (
            merged[col_home] if col_home in merged.columns else pd.NA,
            merged[col_away] if col_away in merged.columns else pd.NA
        )

    # Identity
    out["match_date"] = merged["date"]
    out["season"] = merged["season_home"]
    out["competition"] = merged["comp_home"]
    out["matchweek"] = merged["matchweek_num_home"] if "matchweek_num_home" in merged.columns else merged["round_home"]
    out["home_team"] = merged["team_home"]
    out["away_team"] = merged["team_away"]
    out["referee"] = merged["referee_home"]

    # Goals
    out["home_goals"] = merged["gf_home"]
    out["away_goals"] = merged["ga_home"]
    out["goal_difference"] = merged["gf_home"] - merged["ga_home"]

    # xG
    out["home_xg"] = merged["xg_home"]
    out["away_xg"] = merged["xg_away"]

    # Non-penalty xG
    (h, a) = safe("npxg_home", "npxg_away")
    out["home_npxg"] = h
    out["away_npxg"] = a

    # xGA
    out["home_xga"] = merged["xga_home"]
    out["away_xga"] = merged["xga_away"]

    # Post-shot xG
    (h, a) = safe("psxg_home", "psxg_away")
    out["home_psxg"] = h
    out["away_psxg"] = a

    # Shooting
    out["home_shots"] = merged["sh_home"]
    out["away_shots"] = merged["sh_away"]
    out["home_shots_on_target"] = merged["sot_home"]
    out["away_shots_on_target"] = merged["sot_away"]

    # Avg shot distance
    out["home_avg_shot_distance"] = merged["dist_home"]
    out["away_avg_shot_distance"] = merged["dist_away"]

    # Free kicks
    out["home_free_kicks"] = merged["fk_home"]
    out["away_free_kicks"] = merged["fk_away"]

    # Penalties
    out["home_penalties_scored"] = merged["pk_home"]
    out["away_penalties_scored"] = merged["pk_away"]
    out["home_penalties_attempted"] = merged["pkatt_home"]
    out["away_penalties_attempted"] = merged["pkatt_away"]

    # Defensive: shots on target conceded
    (h, a) = safe("sota_home", "sota_away")
    out["home_sota"] = h
    out["away_sota"] = a

    # Saves
    (h, a) = safe("saves_home", "saves_away")
    out["home_saves"] = h
    out["away_saves"] = a

    # Save %  (CAN BE MISSING IN RAW DATA)
    (h, a) = safe("save_pct_home", "save_pct_away")
    out["home_save_pct"] = h
    out["away_save_pct"] = a

    # Possession
    (h, a) = safe("poss_home", "poss_away")
    out["home_possession"] = h
    out["away_possession"] = a

    # Formations
    out["home_formation"] = merged["formation_home"]
    out["away_formation"] = merged["formation_away"]

    # Final sorting
    out = out.sort_values(["season", "match_date"])

    return out



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("Loading base…")
    df = load_base()

    df = keep_useful(df)
    df = sort_data(df)
    df = build_match_id(df)

    df.to_csv(OUT_BASE, index=False)
    print("✔ Saved base dataset →", OUT_BASE)

    home, away = split_home_away(df)

    print("Home rows :", len(home))
    print("Away rows :", len(away))
    print("Match IDs:", df["match_id"].nunique())

    clean = pivot(home, away)

    clean.to_csv(OUT_PIVOT, index=False)
    print("\n✔ FINAL DATASET SAVED →", OUT_PIVOT)
    print("Final shape:", clean.shape)
