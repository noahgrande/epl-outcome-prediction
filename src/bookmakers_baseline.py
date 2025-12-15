import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix


def build_match_level_from_team(df):
    """
    Reconstruit un dataset match-level à partir du team-level.
    """

    matches = []

    for match_id, g in df.groupby("match_id"):

        if len(g) != 2:
            continue  # sécurité

        home = g[g["is_home"] == 1].iloc[0]
        away = g[g["is_home"] == 0].iloc[0]

        # Résultat réel
        if home["goals_for"] > away["goals_for"]:
            target = 1
        elif home["goals_for"] < away["goals_for"]:
            target = -1
        else:
            target = 0

        matches.append({
            "match_id": match_id,
            "odds_home": home["odds_win"],
            "odds_draw": home["odds_draw"],
            "odds_away": home["odds_lose"],
            "target": target
        })

    return pd.DataFrame(matches)


def bookmaker_accuracy(df_match):
    """
    Accuracy bookmaker sur dataset match-level.
    """

    # Probabilités implicites
    p_home = 1 / df_match["odds_home"]
    p_draw = 1 / df_match["odds_draw"]
    p_away = 1 / df_match["odds_away"]

    # Normalisation
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    # Prédiction bookmaker
    preds = []
    for h, d, a in zip(p_home, p_draw, p_away):
        if h >= d and h >= a:
            preds.append(1)
        elif d >= h and d >= a:
            preds.append(0)
        else:
            preds.append(-1)

    acc = accuracy_score(df_match["target"], preds)

    print("📊 Bookmaker accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(df_match["target"], preds))

    return acc


if __name__ == "__main__":

    DATA_PATH = Path("data/processed/data_before_engineering.csv")

    print("📥 Loading team-level data...")
    df_team = pd.read_csv(DATA_PATH)

    print("⚙️ Building match-level dataset...")
    df_match = build_match_level_from_team(df_team)

    print("📊 Number of matches:", len(df_match))

    bookmaker_accuracy(df_match)




