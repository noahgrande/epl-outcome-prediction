import pandas as pd
from pathlib import Path


def build_match_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le dataset match-level (1 ligne = 1 match)
    en calculant les différences Home - Away.
    """

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    # --------------------------------------------------
    # Colonnes de base (match-level, côté HOME)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Colonnes à différencier (Home - Away)
    # --------------------------------------------------
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

        # 🔥 NOUVELLES FEATURES COMPOSITES
        "avg_goal_diff_L5",
        "avg_xg_diff_L5",
        "avg_discipline_L5",
    ]

    # --------------------------------------------------
    # Merge Home / Away
    # --------------------------------------------------
    merged = home.merge(
        away,
        on="match_id",
        suffixes=("_home", "_away"),
        how="inner"
    )

    # --------------------------------------------------
    # Dataset final
    # --------------------------------------------------
    out = pd.DataFrame({
        "match_id": merged["match_id"]
    })

    # Colonnes match-level (côté HOME)
    for c in base_cols:
        if c == "match_id":
            continue
        out[c] = merged[f"{c}_home"]

    # Différences Home - Away
    for c in feature_cols:
        out[f"diff_{c}"] = (
            merged[f"{c}_home"] - merged[f"{c}_away"]
        )

    # --------------------------------------------------
    # Tri final (sécurité temporelle)
    # --------------------------------------------------
    out = out.sort_values("match_date").reset_index(drop=True)

    return out


# ======================================================
# EXECUTION
# ======================================================
if __name__ == "__main__":

    INPUT_PATH = Path("data/processed/data_team_with_features.csv")
    OUTPUT_PATH = Path("data/processed/data_match_features.csv")

    print("📥 Loading team-level data:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    df["match_date"] = pd.to_datetime(df["match_date"])

    print("⚙️ Building match-level features (Home - Away)...")
    df_match = build_match_level_features(df)

    print("📊 Final shape:", df_match.shape)
    print("\n🔎 Sample output:")
    print(df_match.head())

    df_match.to_csv(OUTPUT_PATH, index=False)
    print("✅ Saved output to:", OUTPUT_PATH)

