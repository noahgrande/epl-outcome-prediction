import pandas as pd
from pathlib import Path


def build_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rolling features par équipe (team-level),
    réinitialisées à chaque saison,
    en utilisant UNIQUEMENT les matchs passés (shift(1)).
    """

    df = df.copy()

    # --------------------------------------------------
    # Sécurité : tri strict pour éviter toute fuite
    # --------------------------------------------------
    df = df.sort_values(
        ["season", "team", "match_date", "match_id"]
    ).reset_index(drop=True)

    # 🔥 RESET PAR SAISON + ÉQUIPE
    g = df.groupby(["season", "team"])

    # --------------------------------------------------
    # 1) Résultats & forme
    # --------------------------------------------------
    df["avg_points_L5"] = g["points"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_points_L10"] = g["points"].shift(1).rolling(10, min_periods=1).mean()

    df["avg_goals_for_L5"] = g["goals_for"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_goals_against_L5"] = g["goals_against"].shift(1).rolling(5, min_periods=1).mean()

    df["clean_sheet_rate_L5"] = g["clean_sheets"].shift(1).rolling(5, min_periods=1).mean()

    # Différence de buts (forme réelle)
    df["avg_goal_diff_L5"] = (
        df["avg_goals_for_L5"] - df["avg_goals_against_L5"]
    )


    # --------------------------------------------------
    # 2) Qualité de jeu (xG)
    # --------------------------------------------------
    df["avg_xg_for_L5"] = g["xg_for"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_xg_against_L5"] = g["xg_against"].shift(1).rolling(5, min_periods=1).mean()
    # Différence de xG
    df["avg_xg_diff_L5"] = (
        df["avg_xg_for_L5"] - df["avg_xg_against_L5"]
    )  
    

    # --------------------------------------------------
    # 3) Pression offensive / défensive
    # --------------------------------------------------
    df["avg_shots_on_target_for_L5"] = (
        g["shots_on_target_for"].shift(1).rolling(5, min_periods=1).mean()
    )

    df["avg_shots_on_target_against_L5"] = (
        g["shots_on_target_against"].shift(1).rolling(5, min_periods=1).mean()
    )

    # --------------------------------------------------
    # 4) Contrôle & gardien
    # --------------------------------------------------
    df["avg_possession_L5"] = g["possession"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_saves_L5"] = g["saves"].shift(1).rolling(5, min_periods=1).mean()

    # --------------------------------------------------
    # 5) Discipline
    # --------------------------------------------------
    df["avg_fouls_L5"] = g["fouls"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_yellow_cards_L5"] = g["yellow_cards"].shift(1).rolling(5, min_periods=1).mean()
    # Discipline globale
    df["avg_discipline_L5"] = (
        df["avg_fouls_L5"] + df["avg_yellow_cards_L5"]
    )


    # --------------------------------------------------
    # 6) Défense active
    # --------------------------------------------------
    df["avg_blocks_L5"] = g["blocks"].shift(1).rolling(5, min_periods=1).mean()
    df["avg_clearances_L5"] = g["clearances"].shift(1).rolling(5, min_periods=1).mean()

    # --------------------------------------------------
    # 7) Home / Away split (points uniquement)
    # --------------------------------------------------
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
# EXECUTION DU SCRIPT
# ======================================================
if __name__ == "__main__":

    INPUT_PATH = Path("data/processed/data_before_engineering.csv")
    OUTPUT_PATH = Path("data/processed/data_after_engineering.csv")

    print("📥 Loading data:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    df["match_date"] = pd.to_datetime(df["match_date"])

    print("⚙️ Building rolling features (team level)...")
    df_features = build_team_rolling_features(df)

    print("📊 Shape before:", df.shape)
    print("📊 Shape after :", df_features.shape)

    print("\n🔎 Sample output:")
    print(df_features.head(10))

    df_features.to_csv(OUTPUT_PATH, index=False)
    print("✅ Saved output to:", OUTPUT_PATH)

