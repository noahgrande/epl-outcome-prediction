from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay

def save_confusion_matrix_png(cm, labels, title, out_path, display_labels=None):
    """
    Save a confusion matrix figure to disk as a PNG.
    This helper centralizes plotting logic so that all models (and the bookmaker
    baseline) produce comparable, consistently formatted confusion matrix figures
    for reporting and reproducibility.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels or labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)



def evaluate_bookmaker(df: pd.DataFrame):
    """
    Evaluate a bookmaker baseline using implied probabilities from odds.

    Odds are converted to implied probabilities and normalized per match.
    This provides a strong, realistic reference point to contextualize ML
    performance (accuracy and probabilistic calibration via log-loss).

    Args:
        df (pd.DataFrame): Match-level dataset containing odds and the true target.
            Required columns: odds_win, odds_draw, odds_lose, target.

    Returns:
        dict: Summary metrics for the bookmaker baseline with keys:
            - "accuracy" (float)
            - "log_loss" (float)
    """

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

    probs_ordered = probs[["book_away", "book_draw", "book_home"]]

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    ll = log_loss(y_true, probs_ordered, labels=[-1, 0, 1])

    save_confusion_matrix_png(
        cm=cm,
        labels=[-1, 0, 1],
        display_labels=["Away win", "Draw", "Home win"],
        title="Confusion Matrix — Bookmaker Baseline (Test Set)",
        out_path="results/visualisation/confusion_matrix_bookmaker.png",
    )


    report = classification_report(
        y_true,
        y_pred,
        labels=[-1, 0, 1],
        target_names=["Away win", "Draw", "Home win"],
        zero_division=0,
    )
    
    print("\n BOOKMAKER BASELINE (NO TRAINING)")
    print("Accuracy:", acc)
    print("Log-loss:", ll)
    print("Confusion matrix:\n", cm)
    print(report)
    
    report_path = Path("results/bookmaker_baseline_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BOOKMAKER BASELINE (NO TRAINING)\n")
        f.write("================================\n\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Log-loss: {ll:.3f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(f"{cm}\n\n")
        f.write(report)
        if not report.endswith("\n"):
            f.write("\n")


    print("Bookmaker baseline report saved to results/bookmaker_baseline_report.txt")

    return {
        "accuracy": acc,
        "log_loss": ll,
    }


def train_models(df: pd.DataFrame, book_metrics: dict = None):
    """
    Train and evaluate ML classifiers on engineered match-level features.

    The function performs a chronological train/test split (to better reflect
    real-world forecasting), trains a scaled Logistic Regression pipeline and
    a Random Forest model, and saves evaluation artifacts (reports, confusion
    matrices, feature importances/coefficients, trained model files).

    If bookmaker metrics are provided, it also produces a unified final summary
    to compare ML models against the bookmaker baseline.

    Args:
        df (pd.DataFrame): Match-level modeling dataset containing diff_* features
            and a multiclass target.
            Expected:
              - Features: columns starting with "diff_"
              - Target: "target" with values in {-1, 0, 1}
              - Optional: "match_date" used to sort chronologically
        book_metrics (dict | None): Optional bookmaker baseline metrics as returned
            by evaluate_bookmaker(), used to include baseline performance in the
            final summary.

    Returns:
        tuple: (log_reg_model, rf_model, metrics)
            - log_reg_model: sklearn Pipeline (StandardScaler + LogisticRegression)
            - rf_model: RandomForestClassifier
            - metrics (dict): Nested dictionary with per-model metrics (accuracy,
              log_loss, classes).
    """

    df = df.copy()
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("match_date").reset_index(drop=True)
    
    x = df.filter(regex="^diff_")
    y = df["target"]
    valid = x.notna().all(axis=1) & y.notna()
    x = x.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    
    split_idx = int(len(x) * 0.8)

    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

    metrics = {}
    
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        ))
    ])

    log_reg.fit(x_train, y_train)

    coef = log_reg.named_steps["clf"].coef_
    feature_names = x.columns.tolist()
    classes = log_reg.named_steps["clf"].classes_

    summary_path = Path("results/logistic_regression_coefficients.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write("LOGISTIC REGRESSION COEFFICIENTS\n")
        f.write("================================\n\n")

        for class_idx, class_label in enumerate(classes):
            f.write(f"Class {class_label} vs others\n")
            f.write("-" * 40 + "\n")

            for feat, weight in zip(feature_names, coef[class_idx]):
                f.write(f"{feat:40s} {weight:+.4f}\n")

            f.write("\n")

    print("Logistic regression summary saved to results/logistic_regression_coefficients.txt")

    y_pred_log = log_reg.predict(x_test)
    y_proba_log = log_reg.predict_proba(x_test)

    log_accuracy = accuracy_score(y_test, y_pred_log)
    log_cm = confusion_matrix(y_test, y_pred_log)
    log_ll = log_loss(
        y_test,
        y_proba_log,
        labels=log_reg.named_steps["clf"].classes_
    )

    save_confusion_matrix_png(
        cm=log_cm,
        labels=[-1, 0, 1],
        display_labels=["Away win", "Draw", "Home win"],
        title="Confusion Matrix — Logistic Regression (Test Set)",
        out_path="results/visualisation/confusion_matrix_logistic_regression.png",
    )


    log_report = classification_report(
        y_test,
        y_pred_log,
        zero_division=0
    )

    print("\n Logistic Regression")
    print("Accuracy:", log_accuracy)
    print("Log-loss:", log_ll)
    print("Confusion matrix:\n", log_cm)
    print(log_report)

    log_report_path = Path("results/logistic_regression_report.txt")
    log_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_report_path, "w") as f:
        f.write("LOGISTIC REGRESSION\n")
        f.write("================================\n\n")

        f.write(f"Accuracy: {log_accuracy}\n")
        f.write(f"Log-loss: {log_ll}\n\n")

        f.write("Confusion matrix:\n")
        f.write(str(log_cm))
        f.write("\n\n")

        f.write(log_report)
        f.write("\n")
        

    print("Logistic Regression report saved to results/logistic_regression_report.txt")

    
    metrics["log_reg"] = {
        "accuracy": accuracy_score(y_test, y_pred_log),
        "log_loss": log_loss(y_test, y_proba_log, labels=log_reg.named_steps["clf"].classes_),
        "classes": list(log_reg.named_steps["clf"].classes_),
    }
    
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(x_train, y_train)
    
    importances = rf.feature_importances_
    feature_names = x.columns.tolist()

    rf_summary_path = Path("results/random_forest_feature_importance.txt")
    rf_summary_path.parent.mkdir(parents=True, exist_ok=True)

    importance_table = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    with open(rf_summary_path, "w") as f:
        f.write("RANDOM FOREST FEATURE IMPORTANCE\n")
        f.write("================================\n\n")
        f.write("Feature                              Importance\n")
        f.write("-----------------------------------------------\n")

        for feat, imp in importance_table:
            f.write(f"{feat:35s} {imp:.4f}\n")

    print("Random Forest feature importance saved to results/random_forest_feature_importance.txt")

    y_pred_rf = rf.predict(x_test)
    y_proba_rf = rf.predict_proba(x_test)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_cm = confusion_matrix(y_test, y_pred_rf)
    rf_ll = log_loss(
        y_test,
        y_proba_rf,
        labels=rf.classes_
    )

    save_confusion_matrix_png(
        cm=rf_cm,
        labels=[-1, 0, 1],
        display_labels=["Away win", "Draw", "Home win"],
        title="Confusion Matrix — Random Forest (Test Set)",
        out_path="results/visualisation/confusion_matrix_random_forest.png",
    )
    rf_report = classification_report(
        y_test,
        y_pred_rf,
        zero_division=0
    )

    print("\n Random Forest")
    print("Accuracy:", rf_accuracy)
    print("Log-loss:", rf_ll)
    print("Confusion matrix:\n", rf_cm)
    print(rf_report)
    
    rf_report_path = Path("results/random_forest_report.txt")
    rf_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(rf_report_path, "w") as f:

        f.write("RANDOM FOREST\n")
        f.write("================================\n\n")

        f.write(f"Accuracy: {rf_accuracy}\n")
        f.write(f"Log-loss: {rf_ll}\n\n")

        f.write("Confusion matrix:\n")
        f.write(str(rf_cm))
        f.write("\n\n")

        f.write(rf_report)
        f.write("\n")

    print("Random Forest report saved to results/random_forest_report.txt")


    metrics["rf"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "log_loss": log_loss(y_test, y_proba_rf, labels=rf.classes_),
        "classes": list(rf.classes_),
    }


    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(log_reg, models_dir / "logistic_regression.pkl")
    joblib.dump(rf, models_dir / "random_forest.pkl")

    print("\nModels saved to /models/")
    

    features_path = models_dir / "feature_list.txt"

    with open(features_path, "w") as f:
        f.write("FEATURES USED FOR TRAINING\n")
        f.write("==========================\n\n")
        for feat in x.columns:
            f.write(f"{feat}\n")

    print("Feature list saved to models/features_list.txt")

    # ======================================================
    # FINAL RESULTS SUMMARY (MODELS + BOOKMAKER)
    # ======================================================

    summary_rows = {
        "Logistic Regression": {
            "Accuracy": metrics["log_reg"]["accuracy"],
            "Log-loss": metrics["log_reg"]["log_loss"],
        },
        "Random Forest": {
            "Accuracy": metrics["rf"]["accuracy"],
            "Log-loss": metrics["rf"]["log_loss"],
        },
    }

    if book_metrics is not None:
        summary_rows["Bookmaker Baseline"] = {
            "Accuracy": book_metrics["accuracy"],
            "Log-loss": book_metrics["log_loss"],
        }

    summary_df = pd.DataFrame.from_dict(summary_rows, orient="index")

    summary_text = (
        "\n\n==============================\n"
        "FINAL RESULTS SUMMARY\n"
        "==============================\n\n"
        + summary_df.round(4).to_string()
        + "\n"
    )

    print(summary_text)

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/final_results_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)


    return log_reg, rf, metrics
    
