from pathlib import Path
import re
import pandas as pd


# ======================================================
# build_matchdata_base.py
# ======================================================

RAW_FILE_MATCHDATA = "data/raw/matchdata_21-25.csv"
OUT_FILE_MATCHDATA = "data/processed/matchdata_base.csv"


team_name_normalization = {
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
    """
    Normalize a team name to a canonical format.

    This reduces inconsistencies across data sources (e.g., "Man Utd", "MU",
    "Manchester United") to make merges and aggregations reliable.

    Args:
        x: Raw team name (any type; will be cast to string).

    Returns:
        str: Normalized, canonical team name in lowercase.
    """
    x = str(x).lower().strip()
    return team_name_normalization.get(x, x)

def normalize_formation(f):
    """
    Normalize a football formation string into a consistent dashed format.

    Handles multiple input formats (e.g., "4-2-3-1", "4/2/3/1", mixed text),
    aiming to reduce noise in categorical features.
    """

    if pd.isna(f):
        return f

    f = str(f).strip()

    if "-" in f and "/" not in f:
        return f

    nums = re.findall(r"(\d+)", f)

    if len(nums) < 2:
        return f

    if len(nums) == 3:
        return f"{nums[0]}-{nums[1]}-{nums[2][0]}"

    if len(nums) == 2:
        return f"{nums[0]}-{nums[1]}"

    return "-".join(nums[:3])

def norm_referee(x):
    """
    Normalize referee names to a consistent full-name format.

    This reduces merge issues and duplicate categories caused by inconsistent
    abbreviations, punctuation, and capitalization across sources.

    Args:
        x: Referee name value (may be NaN).

    Returns:
        Normalized referee name (NaN preserved).
    """
    
    if pd.isna(x):
        return x

    x = str(x).strip()
 
    x = x.replace(".", "").strip()

    x = x.title()

    ref_map = {
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

    if x in ref_map:
        return ref_map[x]

    return x

column_rename = {
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

useful_cols = [
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

def load_raw():
    """
    Load and clean the raw match-level team statistics dataset.

    The function standardizes column names, parses dates robustly, normalizes
    team/opponent names, referee names, and formation strings, and keeps only
    a curated set of useful columns. It also extracts a numeric matchweek to
    ensure correct chronological sorting.

    Returns:
        pd.DataFrame: Cleaned dataset sorted by season, matchweek, team, and date.
    """
    df = pd.read_csv(RAW_FILE_MATCHDATA)

    df.columns = (
        df.columns.str.lower()
                  .str.strip()
                  .str.replace("%", "pct")
                  .str.replace(" ", "_")
                  .str.replace(".", "_")
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    bad = df["date"].isna()
    if bad.any():
        df.loc[bad, "date"] = pd.to_datetime(df.loc[bad, "date"], errors="coerce", dayfirst=True)

    df["team"] = df["team"].apply(normalize_team)
    df["opponent"] = df["opponent"].apply(normalize_team)

    df["referee"] = df["referee"].apply(norm_referee)

    df = df.rename(columns=column_rename)

    for col in ["team_formation", "opponent_formation"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_formation)

    df["matchweek_num"] = df["matchweek"].str.extract(r"(\d+)").astype(int)

    keep = [c for c in useful_cols if c in df.columns]
    df = df[keep + ["matchweek_num"]]

    df = df.sort_values(["season", "matchweek_num", "team", "match_date"])

    return df
#----------------------------------
#BUILD_MATCHDATA_CLEAN
#----------------------------------
BASE_FILE = "data/processed/matchdata_base.csv"
OUT_FILE = "data/processed/matchdata_clean.csv"

def load_base():
    """
    Load the pre-built base dataset and ensure correct date typing.

    This function assumes the base dataset has already been cleaned and saved,
    and enforces datetime parsing to prevent downstream sorting/feature issues.

    Returns:
        pd.DataFrame: Base dataset with match_date as datetime.
    """
    df = pd.read_csv(BASE_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df

    
def prepare_home_away(df):
    """
    Split team-level rows into home and away subsets and create a shared match_id.

    The source dataset contains one row per team per match (two rows per match).
    We generate a stable match identifier based on date + (home_team, away_team)
    so that home and away rows can be merged into a single match-level record.

    Args:
        df (pd.DataFrame): Team-level dataset containing 'venue', 'match_date',
            'team', and 'opponent' columns.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (home_df, away_df) each with a match_id.
    """
    

    df["is_home"] = df["venue"].str.lower().eq("home")

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    def make_match_id(row, is_home):
        date = str(row["match_date"].date())
        home_team = row["team"] if is_home else row["opponent"]
        away_team = row["opponent"] if is_home else row["team"]

        def norm_team(t):
            return (
                str(t)
                .strip()
                .lower()
                .replace(" ", "_")
            )

        h = norm_team(home_team)
        a = norm_team(away_team)

        return f"{date}_{h}_{a}"


    home["match_id"] = home.apply(lambda r: make_match_id(r, True), axis=1)
    away["match_id"] = away.apply(lambda r: make_match_id(r, False), axis=1)

    return home, away

    
def pivot_matches(home, away):
    """
    Convert home/away team-level tables into a single match-level table.

    This creates one row per match, with separate prefixes for home and away
    metrics (e.g., home_xg, away_xg). This wide format is convenient for
    merging bookmaker odds and for feature engineering at match level.

    Args:
        home (pd.DataFrame): Home-team rows with match_id.
        away (pd.DataFrame): Away-team rows with match_id.

    Returns:
        pd.DataFrame: Match-level dataset (one row per match).
    """

    merged = home.merge(
        away,
        on="match_id",
        suffixes=("_home", "_away"),
        validate="one_to_one"
    )

    out = pd.DataFrame()

    out["match_id"] = merged["match_id"]
    out["match_date"] = merged["match_date_home"]
    out["season"] = merged["season_home"]
    out["matchweek"] = merged["matchweek_home"]
    out["matchweek_num"] = merged["matchweek_num_home"]
    out["referee"] = merged["referee_home"]

    out["home_team"] = merged["team_home"]
    out["away_team"] = merged["team_away"]

    out["home_goals"] = merged["goals_for_home"]
    out["away_goals"] = merged["goals_for_away"]
    out["goal_difference"] = merged["goals_for_home"] - merged["goals_for_away"]

    metrics = [
        "xg", "non_penalty_xg", "xg_against", "post_shot_xg",
        "goals_minus_xg", "post_shot_xg_diff",
        "shots", "shots_on_target", "avg_shot_distance",
        "shots_on_target_against", "saves", "clean_sheets",
        "possession", "free_kicks", "penalties_scored", "penalties_attempted",
        "passes_completed", "passes_attempted", "total_distance_progressed",
        "progressive_distance", "progressive_carries",
        "assists", "expected_assisted_goals", "expected_assists",
        "key_passes", "shot_creating_actions", "goal_creating_actions",
        "miscontrols", "dispossessed", "recoveries",
        "tackles", "tackles_won", "interceptions",
        "defensive_actions", "blocks", "clearances"
    ]

    def safe_copy(prefix, metric):
        src = f"{metric}_{prefix}"
        dst = f"{prefix}_{metric}"

        if src in merged.columns:
            out[dst] = merged[src]
        else:
            out[dst] = None

    for m in metrics:
        safe_copy("home", m)
        safe_copy("away", m)

    out["home_formation"] = merged["team_formation_home"]
    out["away_formation"] = merged["opponent_formation_home"]

    out = out.sort_values(
        ["season", "matchweek_num", "match_date", "home_team"]
    )

    return out

# --------------------------------------------------------
# BUILD_ALL_MATCHES_CLEAN.PY
# --------------------------------------------------------

RAW_DIR = "data/raw"
OUT_FILE = "data/processed/all_matches_clean.csv"

column_map = {
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

keep_cols = list(column_map.keys())

    
def load_file(path):
    """
    Load and standardize one raw season file containing match info and bookmaker odds.

    The source files may contain extra columns and inconsistent naming. This function
    keeps only relevant columns, renames them to a common schema, parses dates, and
    normalizes team and referee names to ensure reliable merges later.

    Args:
        path: Path to a raw CSV file.

    Returns:
        pd.DataFrame: Cleaned file-level dataset with standardized column names.
    """
    df = pd.read_csv(path)

    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df = df.rename(columns=column_map)
    
    df["match_date"] = pd.to_datetime(df["match_date"], format="%d/%m/%Y", errors="coerce")
    
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)
    df["referee"] = df["referee"].apply(norm_referee)

    return df


def build_match_id(row):
    """
    Build a stable match identifier from date and teams.

    This key is used to merge match statistics with bookmaker odds, and must be
    consistent across datasets.

    Args:
        row: A DataFrame row with 'match_date', 'home_team', and 'away_team'.

    Returns:
        str: match_id in the format 'YYYY-MM-DD_home_away'.
    """
    return (
        f"{row['match_date'].strftime('%Y-%m-%d')}_"
        f"{row['home_team'].replace(' ', '_').lower()}_"
        f"{row['away_team'].replace(' ', '_').lower()}"
    )

def build_all():
    """
    Build a unified dataset from multiple raw season files (odds/match info).

    Loads selected season files, concatenates them into one table, filters to a
    cutoff date to avoid incomplete periods, and creates a match_id for merging.

    Returns:
        pd.DataFrame: Concatenated dataset with a match_id column.
    """
    raw = Path(RAW_DIR)
    files = sorted([f for f in raw.glob("*.csv") if f.name.startswith(("21_22", "22_23", "23_24", "24_25"))])

    dfs = [load_file(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df = df[df["match_date"] <= pd.to_datetime("2025-01-26")]
    df["match_id"] = df.apply(build_match_id, axis=1)
    return df
    
# ======================================================
# MERGE_DATASET.PY
# ======================================================

MATCHDATA_FILE = "data/processed/matchdata_clean.csv"
ALLMATCHES_FILE = "data/processed/all_matches_clean.csv"
MERGED_OUT_FILE = "data/processed/data_merged.csv"

def merge_dataset(matchdata: pd.DataFrame, allm: pd.DataFrame) -> pd.DataFrame:
    """
    Merge match statistics with bookmaker odds using match_id.

    To avoid ambiguous duplicates (same information present in both datasets),
    we drop "duplicate intent" columns from the odds dataset before merging.

    Args:
        matchdata (pd.DataFrame): Match-level statistics dataset (one row per match).
        allm (pd.DataFrame): Odds/match info dataset containing bookmaker columns.

    Returns:
        pd.DataFrame: Merged dataset with odds appended to matchdata.
    """

    duplicate_intent_cols = [
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

    cols_to_keep_allm = [c for c in allm.columns if c not in duplicate_intent_cols]
    allm = allm[cols_to_keep_allm]

    merged = matchdata.merge(allm, on="match_id", how="left")

    return merged
    
# ======================================================
# BUILD_DATA_BEFORE_ENG.PY
# ======================================================

def build_data_before_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level data into a team-level table suitable for rolling features.

    The output has one row per team per match (two rows per match). This format
    makes it straightforward to compute past-performance features (rolling means)
    per team while preventing target leakage by shifting values.

    Args:
        df (pd.DataFrame): Match-level dataset containing home/away stats and odds.

    Returns:
        pd.DataFrame: Team-level dataset with points and standardized columns.
    """
    df = df.copy()

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    home = pd.DataFrame({
        "match_id": df["match_id"],
        "match_date": df["match_date"],
        "season": df["season"],
        "matchweek_num": df["matchweek_num"],
        "referee": df["referee"],

        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": 1,

        "goals_for": df["home_goals"],
        "goals_against": df["away_goals"],

        "xg_for": df["home_xg"],
        "xg_against": df["home_xg_against"],

        "shots_for": df["home_shots"],
        "shots_on_target_for": df["home_shots_on_target"],
        "shots_on_target_against": df["home_shots_on_target_against"],

        "possession": df["home_possession"],
        "saves": df["home_saves"],

        "clean_sheets": df["home_clean_sheets"],
        "fouls": df["home_fouls"],
        "yellow_cards": df["home_yellow_cards"],

        "blocks": df["home_blocks"],
        "clearances": df["home_clearances"],

        "odds_win": df["odds_avg_home_win"],
        "odds_draw": df["odds_avg_draw"],
        "odds_lose": df["odds_avg_away_win"],
        "odds_over25": df["odds_avg_over25"],
        "odds_under25": df["odds_avg_under25"],
    })

    away = pd.DataFrame({
        "match_id": df["match_id"],
        "match_date": df["match_date"],
        "season": df["season"],
        "matchweek_num": df["matchweek_num"],
        "referee": df["referee"],

        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": 0,

        "goals_for": df["away_goals"],
        "goals_against": df["home_goals"],

        "xg_for": df["away_xg"],
        "xg_against": df["away_xg_against"],

        "shots_for": df["away_shots"],
        "shots_on_target_for": df["away_shots_on_target"],
        "shots_on_target_against": df["away_shots_on_target_against"],

        "possession": df["away_possession"],
        "saves": df["away_saves"],

        "clean_sheets": df["away_clean_sheets"],
        "fouls": df["away_fouls"],
        "yellow_cards": df["away_yellow_cards"],

        "blocks": df["away_blocks"],
        "clearances": df["away_clearances"],

        "odds_win": df["odds_avg_away_win"],
        "odds_draw": df["odds_avg_draw"],
        "odds_lose": df["odds_avg_home_win"],
        "odds_over25": df["odds_avg_over25"],
        "odds_under25": df["odds_avg_under25"],
    })

    out = pd.concat([home, away], ignore_index=True)

    out["points"] = 0
    out.loc[out["goals_for"] > out["goals_against"], "points"] = 3
    out.loc[out["goals_for"] == out["goals_against"], "points"] = 1

    out = out.sort_values(
        ["match_id", "match_date", "team"]
    ).reset_index(drop=True)

    return out


# ======================================================
# BUILD_DATA_AFTER_ENG.PY
# ======================================================

def build_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling performance features per team within each season.

    All rolling metrics are shifted by 1 match to avoid target leakage: the features
    for a given match only depend on matches that occurred before it.

    Args:
        df (pd.DataFrame): Team-level dataset (two rows per match) with points and stats.

    Returns:
        pd.DataFrame: Same dataset with added rolling feature columns.
    """
    df = df.copy()

    df = df.sort_values(
        ["season", "team", "match_date", "match_id"]
    ).reset_index(drop=True)

    g = df.groupby(["season", "team"])

    df["avg_points_L5"] = g["points"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_points_L10"] = g["points"].shift(1).rolling(10, min_periods=1).mean()

    df["avg_goals_for_L5"] = g["goals_for"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_goals_against_L5"] = g["goals_against"].shift(1).rolling(5, min_periods=1).mean()

    df["clean_sheet_rate_L5"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).mean()

    df["avg_goal_diff_L5"] = (
        df["avg_goals_for_L5"] - df["avg_goals_against_L5"]
    )

    df["avg_xg_for_L5"] = g["xg_for"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_xg_against_L5"] = g["xg_against"].shift(1).rolling(5, min_periods=1).mean()

    df["avg_xg_diff_L5"] = (
        df["avg_xg_for_L5"] - df["avg_xg_against_L5"]
    )

    df["avg_shots_on_target_for_L5"] = (
        g["shots_on_target_for"].shift(1).rolling(5, min_periods=1).mean()
    )

    df["avg_shots_on_target_against_L5"] = (
        g["shots_on_target_against"].shift(1).rolling(5, min_periods=1).mean()
    )

    df["avg_possession_L5"] = g["possession"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_saves_L5"] = g["saves"].shift(1).rolling(5, min_periods=1).mean()

    df["avg_fouls_L5"] = g["fouls"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_yellow_cards_L5"] = g["yellow_cards"].shift(1).rolling(5, min_periods=1).mean()

    df["avg_discipline_L5"] = (
        df["avg_fouls_L5"] + df["avg_yellow_cards_L5"]
    )

    df["avg_blocks_L5"] = g["blocks"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_clearances_L5"] = g["clearances"].shift(1).rolling(5, min_periods=1).mean()

    df["avg_points_home_L5"] = (
        g.apply(
            lambda x: x["points"]
            .where(x["is_home"] == 1)
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
        )
        .reset_index(level=[0, 1], drop=True)
    )

    df["avg_points_away_L5"] = (
        g.apply(
            lambda x: x["points"]
            .where(x["is_home"] == 0)
            .shift(1)
            .rolling(5, min_periods=1)
            .mean()
        )
        .reset_index(level=[0, 1], drop=True)
    )

    return df


# ======================================================
# BUILD_MODEL_DATASET.PY
# ======================================================

def build_match_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final match-level modeling table from team-level rolling features.

    The function merges the home and away rows for each match, constructs
    difference features (home minus away) to represent relative strength, and
    derives a 3-class target label:
        1  = home win
        0  = draw
        -1 = away win

    Args:
        df (pd.DataFrame): Team-level dataset with rolling features and points.

    Returns:
        pd.DataFrame: Match-level modeling dataset with diff_* features and target.
    """
    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    base_cols = [
        "match_id",
        "match_date",
        "season",
        "matchweek_num",
        "referee",
        "odds_win",
        "odds_draw",
        "odds_lose",
        "odds_over25",
        "odds_under25",
    ]

    feature_cols = [
        "avg_points_L5",
        "avg_points_L10",
        "avg_goals_for_L5",
        "avg_goals_against_L5",
        "clean_sheet_rate_L5",
        "avg_xg_for_L5",
        "avg_xg_against_L5",
        "avg_shots_on_target_for_L5",
        "avg_shots_on_target_against_L5",
        "avg_possession_L5",
        "avg_saves_L5",
        "avg_fouls_L5",
        "avg_yellow_cards_L5",
        "avg_blocks_L5",
        "avg_clearances_L5",
        "avg_points_home_L5",
        "avg_points_away_L5",
        "avg_goal_diff_L5",
        "avg_xg_diff_L5",
        "avg_discipline_L5",
    ]

    merged = home.merge(
        away,
        on="match_id",
        suffixes=("_home", "_away"),
        how="inner"
    )

    out = pd.DataFrame({
        "match_id": merged["match_id"]
    })

    for c in base_cols:
        if c == "match_id":
            continue
        out[c] = merged[f"{c}_home"]

    for c in feature_cols:
        out[f"diff_{c}"] = (
            merged[f"{c}_home"] - merged[f"{c}_away"]
        )

    out["target"] = 0
    out.loc[merged["points_home"] > merged["points_away"], "target"] = 1
    out.loc[merged["points_home"] < merged["points_away"], "target"] = -1

    out = out.sort_values("match_date").reset_index(drop=True)

    return out
