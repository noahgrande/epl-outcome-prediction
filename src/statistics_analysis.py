import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_PATH = Path("results/match_probabilities_comparison.csv")

VIS_PATH = Path("results/visualisation")
VIS_PATH.mkdir(parents=True, exist_ok=True)


def load_results():
    """
    Load the probabilistic comparison results between the ML model
    and the bookmaker baseline.

    This function reads the CSV file produced by the probabilistic
    evaluation step, which contains predicted probabilities from
    the model, implied probabilities from bookmakers, and indicators
    of whether the model assigns a higher probability to the realized
    outcome.

    Raises:
        FileNotFoundError: If the probabilistic comparison file does not exist.

    Returns:
        pandas.DataFrame: DataFrame containing match-level probabilistic
        predictions and comparison metrics.
    """
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "match_probabilities_comparison.csv not found. "
            "Run probabilistic evaluation first."
        )
    return pd.read_csv(RESULTS_PATH)


def stat_model_vs_bookmaker_rate(df):
    """
    Percentage of matches where the model assigns a higher probability
    than the bookmaker to the realized outcome.
    """
    rate = df["model_beats_bookmaker"].mean()
    print("\n📊 STAT 1 — Model vs Bookmaker (confidence on real outcome)")
    print(f"Model > bookmaker on real outcome: {rate*100:.1f}%")
    return rate


def stat_result_distribution_when_model_wins(df):
    """
    Distribution of real outcomes when the model is more confident
    than the bookmaker.
    """
    subset = df[df["model_beats_bookmaker"]]

    mapping = {-1: "Away win", 0: "Draw", 1: "Home win"}
    distribution = subset["target"].map(mapping).value_counts(normalize=True)

    print("\n📊 STAT 2 — Result distribution when model > bookmaker")
    print(distribution * 100)
    return distribution


def stat_model_advantage_by_result(df):
    """
    For each real outcome, percentage of matches where the model
    is more confident than the bookmaker.
    """
    print("\n📊 STAT 3 — Model advantage by real outcome")

    results = {}
    for label, name in [(-1, "Away win"), (0, "Draw"), (1, "Home win")]:
        df_label = df[df["target"] == label]
        rate = df_label["model_beats_bookmaker"].mean()
        results[name] = rate
        print(f"{name}: {rate*100:.1f}%")

    return results


def stat_average_probability_difference(df):
    """
    Average probability difference (model - bookmaker) on the real outcome.
    """

    def proba_diff(row):
        if row["target"] == 1:
            return row["model_home_win"] - row["book_home"]
        elif row["target"] == 0:
            return row["model_draw"] - row["book_draw"]
        else:
            return row["model_away_win"] - row["book_away"]

    df = df.copy()
    df["proba_diff"] = df.apply(proba_diff, axis=1)

    avg_diff = df["proba_diff"].mean()
    avg_diff_when_win = df[df["model_beats_bookmaker"]]["proba_diff"].mean()

    print("\n📊 STAT 4 — Average probability difference (model − bookmaker)")
    print(f"Overall average diff: {avg_diff:.3f}")
    print(f"Average diff when model > bookmaker: {avg_diff_when_win:.3f}")

    return avg_diff, avg_diff_when_win

def plot_result_distribution_when_model_wins(df):
    subset = df[df["model_beats_bookmaker"]]

    mapping = {-1: "Away win", 0: "Draw", 1: "Home win"}
    counts = subset["target"].map(mapping).value_counts(normalize=True)

    plt.figure()
    counts.mul(100).plot(kind="bar")
    plt.ylabel("Percentage (%)")
    plt.title("Result distribution when model > bookmaker")
    plt.tight_layout()
    plt.savefig(
        VIS_PATH/ "result_distribution_when_model_beats_bookmaker.png",
        bbox_inches="tight"
    )
    plt.close()

def plot_model_advantage_by_result(df):
    results = {}

    for label, name in [(-1, "Away win"), (0, "Draw"), (1, "Home win")]:
        df_label = df[df["target"] == label]
        results[name] = df_label["model_beats_bookmaker"].mean() * 100

    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.ylabel("Percentage (%)")
    plt.title("Model advantage by real outcome")
    plt.tight_layout()
    plt.savefig(
        VIS_PATH / "model_advantage_by_real_outcome.png",
        bbox_inches="tight"
    )

def plot_probability_difference_distribution(df):

    def proba_diff(row):
        if row["target"] == 1:
            return row["model_home_win"] - row["book_home"]
        elif row["target"] == 0:
            return row["model_draw"] - row["book_draw"]
        else:
            return row["model_away_win"] - row["book_away"]

    df = df.copy()
    df["proba_diff"] = df.apply(proba_diff, axis=1)

    plt.figure()
    plt.hist(df["proba_diff"], bins=30)
    plt.xlabel("Model probability − Bookmaker probability")
    plt.ylabel("Number of matches")
    plt.title("Distribution of probability differences")
    plt.tight_layout()
    plt.savefig(
        VIS_PATH / "probability_difference_distribution.png",
        bbox_inches="tight"
    )



def run_stats():
    print("Loading probabilistic comparison results...")
    df = load_results()

    stat_model_vs_bookmaker_rate(df)
    stat_result_distribution_when_model_wins(df)
    stat_model_advantage_by_result(df)
    stat_average_probability_difference(df)
    plot_result_distribution_when_model_wins(df)
    plot_model_advantage_by_result(df)
    plot_probability_difference_distribution(df)

    print("\nStatistical analysis completed")


if __name__ == "__main__":
    main()
