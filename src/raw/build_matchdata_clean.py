"""
BUILD MATCHDATA CLEAN — FINAL PIVOT VERSION
Uses the exact list of columns provided.
"""

import pandas as pd

BASE_FILE = "data/processed/matchdata_base.csv"
OUT_FILE = "data/processed/matchdata_clean.csv"


# --------------------------------------------------------
# LOAD BASE
# --------------------------------------------------------
def load_base():
    df = pd.read_csv(BASE_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


# --------------------------------------------------------
# BUILD HOME / AWAY VIEWS
# --------------------------------------------------------
def prepare_home_away(df):

    df["is_home"] = df["venue"].str.lower().eq("home")

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    # NEW: match_id aligned with all_matches_clean
    def make_match_id(row, is_home):
        date = str(row["match_date"].date())
        home_team = row["team"] if is_home else row["opponent"]
        away_team = row["opponent"] if is_home else row["team"]

        # normalize team names exactly like all_matches_clean
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


# --------------------------------------------------------
# PIVOT MATCHES INTO SINGLE ROW
# --------------------------------------------------------
def pivot_matches(home, away):

    merged = home.merge(
        away,
        on="match_id",
        suffixes=("_home", "_away"),
        validate="one_to_one"
    )

    out = pd.DataFrame()

    # -----------------------
    # IDENTITY FIELDS
    # -----------------------
    out["match_id"] = merged["match_id"]
    out["match_date"] = merged["match_date_home"]
    out["season"] = merged["season_home"]
    out["matchweek"] = merged["matchweek_home"]
    out["matchweek_num"] = merged["matchweek_num_home"]
    out["referee"] = merged["referee_home"]

    # -----------------------
    # TEAMS
    # -----------------------
    out["home_team"] = merged["team_home"]
    out["away_team"] = merged["team_away"]

    # -----------------------
    # GOALS & RESULT
    # -----------------------
    out["home_goals"] = merged["goals_for_home"]
    out["away_goals"] = merged["goals_for_away"]
    out["goal_difference"] = merged["goals_for_home"] - merged["goals_for_away"]

    # -----------------------
    # LIST OF METRICS TO PIVOT
    # -----------------------
    METRICS = [
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

    # -----------------------
    # SAFE COPY (NO KEYERROR)
    # -----------------------
    def safe_copy(prefix, metric):
        src = f"{metric}_{prefix}"
        dst = f"{prefix}_{metric}"

        if src in merged.columns:
            out[dst] = merged[src]
        else:
            out[dst] = None

    for metric in METRICS:
        safe_copy("home", metric)
        safe_copy("away", metric)

    # -----------------------
    # FORMATIONS
    # -----------------------
    out["home_formation"] = merged["team_formation_home"]
    out["away_formation"] = merged["opponent_formation_home"]

    # -----------------------
    # SORTING
    # -----------------------
    out = out.sort_values(
        ["season", "matchweek_num", "match_date", "home_team"]
    )

    return out


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":

    print("Loading matchdata_base.csv ...")
    df = load_base()

    print(" → Loaded:", df.shape)

    print("\nBuilding HOME / AWAY views...")
    home, away = prepare_home_away(df)
    print("Home rows:", len(home))
    print("Away rows:", len(away))

    print("\nPivoting matches...")
    clean = pivot_matches(home, away)

    clean.to_csv(OUT_FILE, index=False)

    print("\n✔ DONE — Saved CLEAN PIVOT dataset →", OUT_FILE)
    print("Final shape:", clean.shape)
