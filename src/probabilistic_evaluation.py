import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def bookmaker_probabilities(row):
    
    p_home = 1 / row["odds_win"]
    p_draw = 1 / row["odds_draw"]
    p_away = 1 / row["odds_lose"]

    total = p_home + p_draw + p_away

    return {
        "book_home": p_home / total,
        "book_draw": p_draw / total,
        "book_away": p_away / total,
    }


def decode_result(target):

    if target == 1:
        return "Home win"
    elif target == 0:
        return "Draw"
    else:
        return "Away win"


def model_beats_bookmaker(row):

    if row["target"] == 1:      
        return row["model_home_win"] > row["book_home"]
    elif row["target"] == 0:    
        return row["model_draw"] > row["book_draw"]
    else:                       
        return row["model_away_win"] > row["book_away"]


def run_probabilistic_evaluation(
    data_path: Path | str = "data/processed/model_data.csv",
    output_path: Path | str = "results/match_probabilities_comparison.csv",
    sample_n: int = 5,
    verbose: bool = True,
):
    
    data_path = Path(data_path)
    output_path = Path(output_path)

    if verbose:
        print("Loading dataset...")
    df = pd.read_csv(data_path)

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    
    x = df.filter(regex="^diff_")
    y = df["target"]

    split_idx = int(len(df) * 0.8)

    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, _ = y.iloc[:split_idx], y.iloc[split_idx:]

    df_test = df.iloc[split_idx:].reset_index(drop=True)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])

    if verbose:
        print("Training model...")
    model.fit(x_train, y_train)
    
    proba = model.predict_proba(x_test)
    
    df_test["model_away_win"] = proba[:, 0]
    df_test["model_draw"] = proba[:, 1]
    df_test["model_home_win"] = proba[:, 2]
    
    book_probs = df_test.apply(bookmaker_probabilities, axis=1, result_type="expand")
    df_test = pd.concat([df_test, book_probs], axis=1)
    
    if verbose:
        print("\nSample match predictions:\n")
        for _, row in df_test.head(sample_n).iterrows():
            real_result = decode_result(row["target"])

            print(f"Match: {row['match_id']}")
            print(f"Real result → {real_result}")
            print(
                f"Model → Home: {row['model_home_win']*100:.1f}% | "
                f"Draw: {row['model_draw']*100:.1f}% | "
                f"Away: {row['model_away_win']*100:.1f}%"
            )
            print(
                f"Book  → Home: {row['book_home']*100:.1f}% | "
                f"Draw: {row['book_draw']*100:.1f}% | "
                f"Away: {row['book_away']*100:.1f}%"
            )
            print("-" * 55)
    
    final_cols = [
        "match_id",
        "match_date",
        "season",
        "matchweek_num",
        "referee",
        "target",
        "model_home_win",
        "model_draw",
        "model_away_win",
        "book_home",
        "book_draw",
        "book_away",
    ]

    df_final = df_test[final_cols].copy()
    
    df_final["model_beats_bookmaker"] = df_test.apply(model_beats_bookmaker, axis=1)

    n_total = len(df_final)
    n_wins = int(df_final["model_beats_bookmaker"].sum())
    win_rate = (n_wins / n_total) if n_total else 0.0

    if verbose:
        print("\n Model vs Bookmaker (probabilistic comparison)")
        print(
            f"Model assigns higher probability than bookmaker on "
            f"{n_wins} / {n_total} matches "
            f"({100 * win_rate:.1f}%)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    if verbose:
        print("\n Saved:", output_path)

    summary = {"n_total": n_total, "n_wins": n_wins, "win_rate": win_rate}
    return df_final, summary
    