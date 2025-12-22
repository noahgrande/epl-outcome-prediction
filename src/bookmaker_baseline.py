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
    
    print("\n BOOKMAKER BASELINE (NO TRAINING)")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[-1, 0, 1],
            target_names=["Away win", "Draw", "Home win"],
            zero_division=0,
        )
    )
    
    probs_ordered = probs[["book_away", "book_draw", "book_home"]]

    ll = log_loss(
        y_true,
        probs_ordered,
        labels=[-1, 0, 1]
    )

    print(f"Log-loss: {ll:.3f}")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": ll,
    }
