import pandas as pd
from pathlib import Path

# ============================================================
# 1. RESULT ENCODING
# ============================================================

def convert_result(df):
    """Add numeric encoding for match result."""
    mapping = {"H": 1, "D": 0, "A": -1}
    df["result_numeric"] = df["result"].map(mapping)
    return df


# ============================================================
# 2. BASIC DIFFERENCES
# ============================================================

def add_basic_differences(df):
    df["shots_difference"] = df["home_shots"] - df["away_shots"]
    df["shots_on_target_difference"] = df["home_shots_on_target"] - df["away_shots_on_target"]
    df["possession_difference"] = df["home_possession"] - df["away_possession"]
    df["xg_difference"] = df["home_xg"] - df["away_xg"]
    df["npxg_difference"] = df["home_non_penalty_xg"] - df["away_non_penalty_xg"]
    df["post_shot_xg_difference"] = df["home_post_shot_xg"] - df["away_post_shot_xg"]
    df["saves_difference"] = df["home_saves"] - df["away_saves"]
    df["clean_sheets_difference"] = df["home_clean_sheets"] - df["away_clean_sheets"]
    return df


# ============================================================
# 3. EFFICIENCY METRICS
# ============================================================

def add_efficiency_features(df):
    df["home_shot_efficiency"] = df["home_goals"] / df["home_shots"].replace(0, pd.NA)
    df["away_shot_efficiency"] = df["away_goals"] / df["away_shots"].replace(0, pd.NA)

    df["home_shot_accuracy"] = df["home_shots_on_target"] / df["home_shots"].replace(0, pd.NA)
    df["away_shot_accuracy"] = df["away_shots_on_target"] / df["away_shots"].replace(0, pd.NA)

    return df


# ============================================================
# 4. BOOKMAKER NORMALIZED PROBABILITIES
# ============================================================

def add_bookmaker_probabilities(df):
    """Convert odds to normalized probabilities and remove raw odds."""
    def odds_to_prob(series):
        return 1 / series.replace(0, pd.NA)

    # Main 1X2 odds
    df["b365_prob_home"] = odds_to_prob(df["odds_b365_home_win"])
    df["b365_prob_draw"] = odds_to_prob(df["odds_b365_draw"])
    df["b365_prob_away"] = odds_to_prob(df["odds_b365_away_win"])

    total = df["b365_prob_home"] + df["b365_prob_draw"] + df["b365_prob_away"]
    df["b365_prob_home_norm"] = df["b365_prob_home"] / total
    df["b365_prob_draw_norm"] = df["b365_prob_draw"] / total
    df["b365_prob_away_norm"] = df["b365_prob_away"] / total

    # Over/Under 2.5
    df["b365_prob_over25"] = odds_to_prob(df["odds_b365_over25"])
    df["b365_prob_under25"] = odds_to_prob(df["odds_b365_under25"])

    total_ou = df["b365_prob_over25"] + df["b365_prob_under25"]
    df["b365_prob_over25_norm"] = df["b365_prob_over25"] / total_ou
    df["b365_prob_under25_norm"] = df["b365_prob_under25"] / total_ou

    # Remove raw bookmaker columns
    raw_cols = [c for c in df.columns if c.startswith("odds_")]
    df = df.drop(columns=raw_cols, errors="ignore")

    return df


# ============================================================
# 5. RANKING SYSTEM (PER SEASON, PER MATCHWEEK)
# ============================================================

def compute_rankings(df):
    """
    Compute EPL ranking per season, per matchweek.
    Ranking is based on cumulative points.
    """
    df = df.sort_values(["season", "match_date"]).copy()

    df["home_points"] = df["result_numeric"].map({1: 3, 0: 1, -1: 0})
    df["away_points"] = df["result_numeric"].map({1: 0, 0: 1, -1: 3})

    df["home_cum"] = df.groupby(["season", "home_team"])["home_points"].cumsum()
    df["away_cum"] = df.groupby(["season", "away_team"])["away_points"].cumsum()

    ranking_rows = []

    for season, season_df in df.groupby("season"):
        for mw, mw_df in season_df.groupby("matchweek_num"):

            table = {}

            for _, row in mw_df.iterrows():
                table[row["home_team"]] = row["home_cum"]
                table[row["away_team"]] = row["away_cum"]

            sorted_table = sorted(table.items(), key=lambda x: x[1], reverse=True)
            ranks = {team: rank + 1 for rank, (team, pts) in enumerate(sorted_table)}

            for _, row in mw_df.iterrows():
                ranking_rows.append({
                    "match_id": row["match_id"],
                    "home_rank_proxy": ranks[row["home_team"]],
                    "away_rank_proxy": ranks[row["away_team"]],
                })

    rank_df = pd.DataFrame(ranking_rows)
    df = df.merge(rank_df, on="match_id", how="left")

    df["rank_difference"] = df["home_rank_proxy"] - df["away_rank_proxy"]

    return df


# ============================================================
# 6. ROLLING LAST 5
# ============================================================

def compute_team_last5(df, team_col):
    rows = []

    for (season, team), g in df.groupby(["season", team_col]):
        g = g.sort_values("match_date").copy()

        g["points"] = g["result_numeric"].map({1: 3, 0: 1, -1: 0})

        g["goals_for"] = g["home_goals"].where(g[team_col] == g["home_team"], g["away_goals"])
        g["goals_against"] = g["away_goals"].where(g[team_col] == g["home_team"], g["home_goals"])

        g["points_last5"] = g["points"].rolling(5).sum()
        g["goals_for_last5"] = g["goals_for"].rolling(5).sum()
        g["goals_against_last5"] = g["goals_against"].rolling(5).sum()
        g["goal_diff_last5"] = g["goals_for_last5"] - g["goals_against_last5"]

        g["wins_last5"] = (g["result_numeric"] == 1).rolling(5).sum()
        g["draws_last5"] = (g["result_numeric"] == 0).rolling(5).sum()
        g["losses_last5"] = (g["result_numeric"] == -1).rolling(5).sum()
        g["form_index"] = 3 * g["wins_last5"] + g["draws_last5"]

        g["shots_temp"] = g["home_shots"].where(g[team_col] == g["home_team"], g["away_shots"])
        g["sot_temp"] = g["home_shots_on_target"].where(g[team_col] == g["home_team"], g["away_shots_on_target"])
        g["xg_temp"] = g["home_xg"].where(g[team_col] == g["home_team"], g["away_xg"])
        g["poss_temp"] = g["home_possession"].where(g[team_col] == g["home_team"], g["away_possession"])

        g["avg_shots_last5"] = g["shots_temp"].rolling(5).mean()
        g["avg_sot_last5"] = g["sot_temp"].rolling(5).mean()
        g["avg_xg_last5"] = g["xg_temp"].rolling(5).mean()
        g["avg_poss_last5"] = g["poss_temp"].rolling(5).mean()

        rows.append(g)

    return pd.concat(rows)


def merge_last5(df):
    home = compute_team_last5(df, "home_team")
    away = compute_team_last5(df, "away_team")

    rename_home = {col: f"home_{col}" for col in home.columns if col.endswith("last5") or col == "form_index"}
    rename_away = {col: f"away_{col}" for col in away.columns if col.endswith("last5") or col == "form_index"}

    home = home.rename(columns=rename_home)
    away = away.rename(columns=rename_away)

    keep_home = ["match_id"] + list(rename_home.values())
    keep_away = ["match_id"] + list(rename_away.values())

    df = df.merge(home[keep_home], on="match_id", how="left")
    df = df.merge(away[keep_away], on="match_id", how="left")

    return df


# ============================================================
# 7. PIPELINE
# ============================================================

def add_all_features(df):
    df["match_date"] = pd.to_datetime(df["match_date"])

    df = convert_result(df)
    df = add_basic_differences(df)
    df = add_efficiency_features(df)
    df = add_bookmaker_probabilities(df)
    df = compute_rankings(df)
    df = merge_last5(df)

    df = df.drop(columns=["home_formation", "away_formation"], errors="ignore")

    return df


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    input_path = "data/processed/data_merged.csv"
    output_path = "data/processed/data_merged_with_features.csv"

    print("Loading dataset…")
    df = pd.read_csv(input_path)
    initial_cols = df.shape[1]

    print("Adding features…")
    df_features = add_all_features(df)
    final_cols = df_features.shape[1]

    print("Saving enriched dataset…")
    df_features.to_csv(output_path, index=False)

    print("\n✔ Feature engineering complete.")
    print(f"Columns before: {initial_cols}")
    print(f"Columns after : {final_cols}")
    print("--------------------------------------")
    print("Dataset saved successfully!\n")






