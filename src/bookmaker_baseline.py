import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
)


def evaluate_bookmaker(df: pd.DataFrame):
    """
    Évalue les bookmakers comme baseline externe :
    - odds → probabilités implicites
    - prédiction = argmax(probabilité)
    - mêmes métriques qu’un modèle ML
    """

    # -----------------------------
    # Sécurité : colonnes requises
    # -----------------------------
    required_cols = ["odds_win", "odds_draw", "odds_lose", "target"]
    df = df.dropna(subset=required_cols).copy()

    # -----------------------------
    # Probabilités implicites
    # -----------------------------
    probs = 1 / df[["odds_win", "odds_draw", "odds_lose"]]
    probs = probs.div(probs.sum(axis=1), axis=0)

    # Renommage explicite
    probs.columns = ["book_home", "book_draw", "book_away"]

    # -----------------------------
    # Prédiction bookmaker (argmax)
    # -----------------------------
    y_pred = probs.idxmax(axis=1).map({
        "book_home": 1,
        "book_draw": 0,
        "book_away": -1,
    })

    y_true = df["target"]

    # -----------------------------
    # Metrics classification
    # -----------------------------
    print("\n💰 BOOKMAKER BASELINE (NO TRAINING)")
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

    # -----------------------------
    # Log-loss (ordre strict)
    # labels = [-1, 0, 1]
    # probs  = [away, draw, home]
    # -----------------------------
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








