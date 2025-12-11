import pandas as pd
from pathlib import Path

RAW_DIR = "data/raw"
OUT_FILE = "data/processed/all_matches_clean.csv"

# ===========================================
# NORMALISATION EQUIPES (même mapping que matchdata_clean)
# ===========================================
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

def norm_team(x):
    if pd.isna(x):
        return x
    x = x.strip().lower()
    return TEAM_NORMALIZATION.get(x, x)

def norm_referee(x):
    """
    Standardise les noms d'arbitres pour correspondre à matchdata_clean :
    - enlève les initiales seules ("M Oliver" → "Michael Oliver")
    - supprime les points ("J. Moss" → "J Moss")
    - garde toujours format Title Case
    - remplace abréviations connues
    """

    if pd.isna(x):
        return x

    x = str(x).strip()

    # enlever les points
    x = x.replace(".", "").strip()

    # mettre en Title Case (ex: "m oliver" -> "M Oliver")
    x = x.title()

    # mapping arbitres EPL (corrige les initiales vers noms complets)
    REFEREE_MAP = {
        "M Oliver": "Michael Oliver",
        "P Tierney": "Paul Tierney",
        "D Coote": "David Coote",
        "J Moss": "Jonathan Moss",
        "A Madley": "Andy Madley",
        "C Pawson": "Craig Pawson",
        "M Dean": "Mike Dean",
        "A Marriner": "Andre Marriner",
        "T Harrington": "Tony Harrington",
        "R Jones": "Robert Jones",
        "S Hooper": "Simon Hooper",
        "J Gillett": "Jarred Gillett",
        "T Robinson": "Tim Robinson",
    }

    if x in REFEREE_MAP:
        return REFEREE_MAP[x]

    return x

# ===========================================
# COLUMNS TO KEEP + RENAME
# ===========================================
COLUMN_MAP = {
    "Date": "match_date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",

    # bookmaker odds
    "B365H": "odds_b365_home_win",
    "B365D": "odds_b365_draw",
    "B365A": "odds_b365_away_win",

    "PSH": "odds_ps_home_win",
    "PSD": "odds_ps_draw",
    "PSA": "odds_ps_away_win",

    "MaxH": "odds_max_home_win",
    "MaxD": "odds_max_draw",
    "MaxA": "odds_max_away_win",

    "AvgH": "odds_avg_home_win",
    "AvgD": "odds_avg_draw",
    "AvgA": "odds_avg_away_win",

    "B365>2.5": "odds_b365_over25",
    "B365<2.5": "odds_b365_under25",
    "Max>2.5": "odds_max_over25",
    "Max<2.5": "odds_max_under25",
    "Avg>2.5": "odds_avg_over25",
    "Avg<2.5": "odds_avg_under25",
}

KEEP_COLS = list(COLUMN_MAP.keys())

# ===========================================
# LOAD & CLEAN ONE SEASON
# ===========================================
def load_file(path):
    df = pd.read_csv(path)

    # keep only useful columns
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

    # rename
    df = df.rename(columns=COLUMN_MAP)

    # parse date
    df["match_date"] = pd.to_datetime(df["match_date"], format="%d/%m/%Y", errors="coerce")

    # normalize teams & refs
    df["home_team"] = df["home_team"].apply(norm_team)
    df["away_team"] = df["away_team"].apply(norm_team)
    df["referee"] = df["referee"].apply(norm_referee)

    return df

# ===========================================
# MAIN BUILD
# ===========================================
def build_match_id(row):
    return (
        f"{row['match_date'].strftime('%Y-%m-%d')}_"
        f"{row['home_team'].replace(' ', '_').lower()}_"
        f"{row['away_team'].replace(' ', '_').lower()}"
    )

def build_all():
    raw = Path(RAW_DIR)
    files = sorted([f for f in raw.glob("*.csv") if f.name.startswith(("21_22", "22_23", "23_24", "24_25"))])

    dfs = [load_file(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # stop at last known FBRef match (Matchweek 23, 2025-01-26)
    df = df[df["match_date"] <= pd.to_datetime("2025-01-26")]
    df["match_id"] = df.apply(build_match_id, axis=1)
    return df


if __name__ == "__main__":
    df = build_all()
    df.to_csv(OUT_FILE, index=False)
    print("✔ all_matches_clean saved:", OUT_FILE)
