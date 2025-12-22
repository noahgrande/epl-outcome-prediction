from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


def evaluate_bookmaker(df: pd.DataFrame):

    required_cols = ["odds_win", "odds_draw", "odds_lose", "target"]
    df = df.dropna(subset=required_cols).copy()

    probs = 1 / df[["odds_win", "odds_draw", "odds_lose"]]
    probs = probs.div(probs.sum(axis=1), axis=0)

    probs.columns = ["book_home", "book_draw", "book_away"]

    y_pred = probs.idxmax(axis=1).map({
        "book_home": 1,
        "book_draw": 0,
        "book_away": -1,
    })

    y_true = df["target"]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=[-1, 0, 1],
        target_names=["Away win", "Draw", "Home win"],
        zero_division=0,
    )

    probs_ordered = probs[["book_away", "book_draw", "book_home"]]
    ll = log_loss(y_true, probs_ordered, labels=[-1, 0, 1])

    print("\n BOOKMAKER BASELINE (NO TRAINING)")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report)
    print(f"Log-loss: {ll:.3f}")

    report_path = Path("results/bookmaker_baseline_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BOOKMAKER BASELINE (NO TRAINING)\n")
        f.write("================================\n\n")
        f.write(f"Accuracy: {acc}\n")
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n\n")
        f.write(report)
        if not report.endswith("\n"):
            f.write("\n")
        f.write(f"Log-loss: {ll:.3f}\n")

    print("Bookmaker baseline report saved to results/bookmaker_baseline_report.txt")

    return {
        "accuracy": acc,
        "log_loss": ll,
    }

