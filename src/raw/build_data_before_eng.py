import pandas as pd
from pathlib import Path


def build_data_before_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    # -------------------------
    # HOME perspective
    # -------------------------
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

        # 🔁 CHANGED — odds AVERAGE (point de vue HOME)
        "odds_win": df["odds_avg_home_win"],
        "odds_draw": df["odds_avg_draw"],
        "odds_lose": df["odds_avg_away_win"],
        "odds_over25": df["odds_avg_over25"],
        "odds_under25": df["odds_avg_under25"],
    })

    # -------------------------
    # AWAY perspective
    # -------------------------
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

        # 🔁 CHANGED — odds AVERAGE (point de vue AWAY)
        "odds_win": df["odds_avg_away_win"],
        "odds_draw": df["odds_avg_draw"],
        "odds_lose": df["odds_avg_home_win"],
        "odds_over25": df["odds_avg_over25"],
        "odds_under25": df["odds_avg_under25"],
    })

    # concat -> 2 lignes par match
    out = pd.concat([home, away], ignore_index=True)

    # points
    out["points"] = 0
    out.loc[out["goals_for"] > out["goals_against"], "points"] = 3
    out.loc[out["goals_for"] == out["goals_against"], "points"] = 1

    # tri final
    out = out.sort_values(
        ["match_id", "match_date", "team"]
    ).reset_index(drop=True)

    return out


if __name__ == "__main__":
    input_path = Path("data/processed/data_merged.csv")
    output_path = Path("data/processed/data_before_engineering.csv")

    df_merge = pd.read_csv(input_path)

    df_before = build_data_before_engineering(df_merge)

    print("Matches (input rows):", len(df_merge))
    print("Team-match rows (output rows):", len(df_before))
    print("Expected output rows ~ 2x:", 2 * len(df_merge))

    df_before.to_csv(output_path, index=False)
    print("Saved:", output_path)











