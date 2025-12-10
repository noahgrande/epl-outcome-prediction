import pandas as pd
from pathlib import Path
import re


# ---------------------------------------------------------
# 1. Load & clean raw matchdata file
# ---------------------------------------------------------
def load_matchdata(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean team names
    df["team"] = df["team"].str.lower().str.strip()
    df["opponent"] = df["opponent"].str.lower().str.strip()

    return df


# ---------------------------------------------------------
# 2. Pivot team-level → match-level (home & away)
# ---------------------------------------------------------

def fix_formation(value):
    """Fixes FBref formations badly encoded (date-like), ensures consistent format."""

    if pd.isna(value):
        return value

    value = str(value).strip()

    # Step 1 : Replace "/" with "-" always
    value = value.replace("/", "-")

    # Step 2 : Split into components
    parts = value.split("-")

    # Remove empty strings
    parts = [p for p in parts if p != ""]

    # Step 3 : Remove last part if it looks like a year (4 digits)
    if len(parts) > 2 and len(parts[-1]) == 4 and parts[-1].isdigit():
        parts = parts[:-1]

    # Step 4 : Convert all remaining parts to integers if possible
    cleaned = []
    for p in parts:
        if p.isdigit():
            cleaned.append(int(p))
        else:
            return value  # If non-numeric → return as is

    # Step 5 : Reconstruct formation based on number of elements
    if len(cleaned) == 4:
        # perfect formation e.g. 4-2-3-1
        return "-".join(str(x) for x in cleaned)

    elif len(cleaned) == 3:
        # classic 3-line formation e.g. 4-3-3
        return "-".join(str(x) for x in cleaned)

    elif len(cleaned) == 2:
        # assume missing last line → infer a "2"
        # e.g. 4-4 → 4-4-2  (common assumption)
        return f"{cleaned[0]}-{cleaned[1]}-2"

    elif len(cleaned) == 1:
        # garbage → return original
        return value

    else:
        return value


def pivot_matchdata(df):

    # HOME rows
    home = df[df["venue"].str.lower() == "home"].copy()
    home = home.rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "gf": "home_goals",
        "ga": "away_goals",
        "xg": "home_xg",
        "npxg": "home_npxg",
        "xga": "home_xga",
        "sh": "home_shots",
        "sot": "home_shots_on_target",
        "dist": "home_avg_shot_distance",
        "fk": "home_free_kicks",
        "pk": "home_penalties_scored",
        "pkatt": "home_penalties_attempted",
        "poss": "home_possession",
        "g-xg": "home_g_minus_xg",
        "sota": "home_sota",
        "saves": "home_saves",
        "save%": "home_save_pct",
        "psxg": "home_psxg",
        "psxg+-": "home_psxg_plus_minus",
        "formation": "home_formation"
    })

    # AWAY rows
    away = df[df["venue"].str.lower() == "away"].copy()
    away = away.rename(columns={
        "team": "away_team",
        "opponent": "home_team",
        "gf": "away_goals",
        "ga": "home_goals",
        "xg": "away_xg",
        "npxg": "away_npxg",
        "xga": "away_xga",
        "sh": "away_shots",
        "sot": "away_shots_on_target",
        "dist": "away_avg_shot_distance",
        "fk": "away_free_kicks",
        "pk": "away_penalties_scored",
        "pkatt": "away_penalties_attempted",
        "poss": "away_possession",
        "g-xg": "away_g_minus_xg",
        "sota": "away_sota",
        "saves": "away_saves",
        "save%": "away_save_pct",
        "psxg": "away_psxg",
        "psxg+-": "away_psxg_plus_minus",
        "formation": "away_formation"
    })

    # Merge home + away
    merged = pd.merge(
        home,
        away,
        on=["date", "home_team", "away_team", "season", "comp", "round", "referee"],
        how="inner",
    )

    # Fix weird formations encoded like dates
    merged["home_formation"] = merged["home_formation"].apply(fix_formation)
    merged["away_formation"] = merged["away_formation"].apply(fix_formation)

    merged = merged.rename(columns={
        "date": "match_date",
        "comp": "competition"
    })
    
    return merged



# ---------------------------------------------------------
# 3. Keep only the useful columns (validated list)
# ---------------------------------------------------------
def select_useful_columns(df):

    keep_cols = [
        # identification
        "match_date", "season", "competition", "round",
        "home_team", "away_team", "referee", "attendance",

        # result
        "home_goals", "away_goals", "result",

        # offensive stats
        "home_xg", "away_xg",
        "home_npxg", "away_npxg",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_avg_shot_distance", "away_avg_shot_distance",
        "home_free_kicks", "away_free_kicks",
        "home_penalties_scored", "away_penalties_scored",
        "home_penalties_attempted", "away_penalties_attempted",
        "home_g_minus_xg", "away_g_minus_xg",

        # defensive stats
        "home_xga", "away_xga",
        "home_sota", "away_sota",
        "home_saves", "away_saves",
        "home_save_pct", "away_save_pct",
        "home_psxg", "away_psxg",
        "home_psxg_plus_minus", "away_psxg_plus_minus",

        # tactical & context
        "home_formation", "away_formation",
        "home_possession", "away_possession"
    ]

    # keep only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols]

    return df


# ---------------------------------------------------------
# 4. Save output
# ---------------------------------------------------------
def save_output(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✔ File saved → {path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    RAW_PATH = "/files/epl-outcome-prediction-1/data/raw/matchdata_21-25.csv"
    OUT_PATH = "/files/epl-outcome-prediction-1/data/processed/matchdata_clean.csv"

    print("Loading dataset…")
    df = load_matchdata(RAW_PATH)

    print("Pivoting into match-level format…")
    df = pivot_matchdata(df)

    print("Selecting useful columns…")
    df = select_useful_columns(df)
    df = df.sort_values(by="match_date").reset_index(drop=True)

    save_output(df, OUT_PATH)

    print("✔ Processing complete.")

