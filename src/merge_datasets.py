"""
Merge bookmaker data with team-level stats (xG, shots, etc.)
Datasets used:
- data/processed/all_matches_clean.csv
- data/raw/matches.csv
- data/raw/matchdata_21-25.csv

Output:
- data/processed/final_dataset.csv
"""

import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Utility: standardize team names
# ---------------------------------------------------------
def standardize_team_name(name: str) -> str:
    """Lowercase, remove weird chars, harmonize some common variants."""
    if pd.isna(name):
        return name
    name = name.strip().lower()

    # Quelques corrections manuelles possibles (tu peux en ajouter)
    replacements = {
        "man united": "manchester united",
        "man utd": "manchester united",
        "man city": "manchester city",
        "nott'ham forest": "nottingham forest",
        "nott'm forest": "nottingham forest",
        "wolves": "wolverhampton",
        "west brom": "west bromwich",
    }
    name = replacements.get(name, name)
    return name


# ---------------------------------------------------------
# 1. Load bookmaker dataset (all_matches_clean)
# ---------------------------------------------------------
def load_bookmaker(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # On suppose que all_matches_clean a ces colonnes:
    # date, hometeam, awayteam, fthg, ftag, ftr, referee, hs, as, hst, ast, b365h, b365d, b365a, psh, psd, psa
    rename_map = {
        "date": "match_date",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "fthg": "home_goals",
        "ftag": "away_goals",
        "ftr": "match_result",
        "hs": "home_shots",
        "as": "away_shots",
        "hst": "home_shots_on_target",
        "ast": "away_shots_on_target",
        "b365h": "bet365_home_odds",
        "b365d": "bet365_draw_odds",
        "b365a": "bet365_away_odds",
        "psh": "pinnacle_home_odds",
        "psd": "pinnacle_draw_odds",
        "psa": "pinnacle_away_odds",
    }
    df = df.rename(columns=rename_map)

    # Dates football-data → dd/mm/yyyy en général
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True, errors="coerce")

    # Standardiser noms d'équipes
    df["home_team"] = df["home_team"].apply(standardize_team_name)
    df["away_team"] = df["away_team"].apply(standardize_team_name)

    return df


# ---------------------------------------------------------
# 2. Load team-level stats from matches.csv
# ---------------------------------------------------------
def load_matches_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Colonnes importantes d'après ton exemple
    rename_map = {
        "date": "match_date",
        "team": "team",
        "opponent": "opponent",
        "season": "season",
        "xg": "xg",
        "xga": "xga",
        "sh": "shots",
        "sot": "shots_on_target",
        "fk": "free_kicks",
        "pk": "pens_scored",
        "pkatt": "pens_att",
        "formation": "formation",
    }
    df = df.rename(columns=rename_map)

    # Garder seulement ce qu'on utilise
    keep_cols = ["match_date", "team", "opponent", "season",
                 "xg", "xga", "shots", "shots_on_target",
                 "free_kicks", "pens_scored", "pens_att",
                 "formation"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Dates type 2020-09-21
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Standardiser noms
    df["team"] = df["team"].apply(standardize_team_name)
    df["opponent"] = df["opponent"].apply(standardize_team_name)

    return df


# ---------------------------------------------------------
# 3. Load team-level stats from matchdata_21-25.csv
# ---------------------------------------------------------
def load_matchdata_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Colonnes importantes dans ce gros fichier
    rename_map = {
        "date": "match_date",
        "team": "team",
        "opponent": "opponent",
        "season": "season",
        "xg": "xg",
        "xga": "xga",
        "sh": "shots",
        "sot": "shots_on_target",
        "fk": "free_kicks",
        "pk": "pens_scored",
        "pkatt": "pens_att",
        "formation": "formation",
        "opp formation": "opp_formation",
        "sota": "shots_on_target_against",
        "saves": "saves",
        "save%": "save_pct",
        "psxg": "post_shot_xg",
    }
    df = df.rename(columns=rename_map)

    keep_cols = [
        "match_date", "team", "opponent", "season",
        "xg", "xga", "shots", "shots_on_target",
        "free_kicks", "pens_scored", "pens_att",
        "formation", "opp_formation",
        "shots_on_target_against", "saves", "save_pct", "post_shot_xg"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Dates dans matchdata: "8/17/2024" → US style, month/day/year
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=False, errors="coerce")

    df["team"] = df["team"].apply(standardize_team_name)
    df["opponent"] = df["opponent"].apply(standardize_team_name)

    return df


# ---------------------------------------------------------
# 4. Combine team-level stats
# ---------------------------------------------------------
def build_team_stats(matches_df: pd.DataFrame, matchdata_df: pd.DataFrame) -> pd.DataFrame:
    """Concatène matches + matchdata en un seul tableau team-level harmonisé."""
    combined = pd.concat([matches_df, matchdata_df], ignore_index=True)

    # On peut enlever les doublons exacts si nécessaire
    combined = combined.drop_duplicates(
        subset=["match_date", "team", "opponent"],
        keep="first"
    )

    return combined


# ---------------------------------------------------------
# 5. Merge team stats into bookmaker dataset
# ---------------------------------------------------------
def merge_team_stats_with_bookmaker(df_bm: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme team_stats en stats home/away puis merge avec df_bm.
    On matche par (match_date, home_team, away_team).
    """

    # HOME: team joue à domicile
    home = team_stats.rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "xg": "home_xg",
        "xga": "home_xga",
        "shots": "home_shots_fbref",
        "shots_on_target": "home_shots_on_target_fbref",
        "free_kicks": "home_free_kicks",
        "pens_scored": "home_penalties_scored",
        "pens_att": "home_penalties_attempted",
        "formation": "home_formation",
        "shots_on_target_against": "home_shots_on_target_against",
        "saves": "home_saves",
        "save_pct": "home_save_pct",
        "post_shot_xg": "home_post_shot_xg",
    })

    home_keep = [
        "match_date", "home_team", "away_team", "season",
        "home_xg", "home_xga",
        "home_shots_fbref", "home_shots_on_target_fbref",
        "home_free_kicks",
        "home_penalties_scored", "home_penalties_attempted",
        "home_formation",
        "home_shots_on_target_against", "home_saves",
        "home_save_pct", "home_post_shot_xg"
    ]
    home = home[[c for c in home_keep if c in home.columns]].drop_duplicates()

    # Merge sur home
    df = df_bm.merge(
        home,
        on=["match_date", "home_team", "away_team"],
        how="left"
    )

    # AWAY: team joue à l'extérieur → on inverse team/opponent
    away = team_stats.rename(columns={
        "team": "away_team",
        "opponent": "home_team",
        "xg": "away_xg",
        "xga": "away_xga",
        "shots": "away_shots_fbref",
        "shots_on_target": "away_shots_on_target_fbref",
        "free_kicks": "away_free_kicks",
        "pens_scored": "away_penalties_scored",
        "pens_att": "away_penalties_attempted",
        "formation": "away_formation",
        "shots_on_target_against": "away_shots_on_target_against",
        "saves": "away_saves",
        "save_pct": "away_save_pct",
        "post_shot_xg": "away_post_shot_xg",
    })

    away_keep = [
        "match_date", "home_team", "away_team", "season",
        "away_xg", "away_xga",
        "away_shots_fbref", "away_shots_on_target_fbref",
        "away_free_kicks",
        "away_penalties_scored", "away_penalties_attempted",
        "away_formation",
        "away_shots_on_target_against", "away_saves",
        "away_save_pct", "away_post_shot_xg"
    ]
    away = away[[c for c in away_keep if c in away.columns]].drop_duplicates()

    df = df.merge(
        away,
        on=["match_date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_dup")
    )

    # Optionnel: on peut drop les eventuels season_dup si ça apparait
    if "season_dup" in df.columns:
        df = df.drop(columns=["season_dup"])

    return df


# ---------------------------------------------------------
# 6. Save dataset
# ---------------------------------------------------------
def save_dataset(df: pd.DataFrame, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Final dataset saved → {path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    BOOKMAKER_FILE = "../data/processed/all_matches_clean.csv"
    MATCHES_FILE = "../data/raw/matches.csv"
    MATCHDATA_FILE = "../data/raw/matchdata_21-25.csv"
    OUT_FILE = "../data/processed/final_dataset.csv"

    print("Loading bookmaker dataset...")
    df_bm = load_bookmaker(BOOKMAKER_FILE)

    print("Loading matches.csv stats...")
    df_matches = load_matches_stats(MATCHES_FILE)

    print("Loading matchdata_21-25.csv stats...")
    df_matchdata = load_matchdata_stats(MATCHDATA_FILE)

    print("Building combined team-level stats...")
    df_team_stats = build_team_stats(df_matches, df_matchdata)

    print("Merging team stats into bookmaker dataset...")
    df_final = merge_team_stats_with_bookmaker(df_bm, df_team_stats)

    save_dataset(df_final, OUT_FILE)
    print("Done.")
