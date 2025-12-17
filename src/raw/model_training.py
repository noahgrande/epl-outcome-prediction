import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_models(df: pd.DataFrame):
    """
    Entraîne une Logistic Regression et un Random Forest
    avec un split temporel (80 / 20).
    """

    # --------------------------------------------------
    # Sélection des features et de la cible
    # --------------------------------------------------
    X = df.filter(regex="^diff_")
    y = df["target"]

    # --------------------------------------------------
    # Split temporel 80 / 20
    # --------------------------------------------------
    split_idx = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # --------------------------------------------------
    # 1) Logistic Regression (avec scaling)
    # --------------------------------------------------
    log_reg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    print("\n📊 Logistic Regression")
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_log))
    print(classification_report(y_test, y_pred_log, zero_division=0))

    # --------------------------------------------------
    # 2) Random Forest (sans scaling)
    # --------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\n🌲 Random Forest")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    return log_reg, rf


# ======================================================
# EXECUTION
# ======================================================
if __name__ == "__main__":

    DATA_PATH = Path("data/processed/model_data.csv")

    print("📥 Loading model dataset...")
    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------
    # Sécurité : tri temporel explicite
    # --------------------------------------------------
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    # --------------------------------------------------
    # Drop des lignes avec NaN dans les features
    # --------------------------------------------------
    df = df.dropna().reset_index(drop=True)


    print("⚙️ Training models...")
    log_model, rf_model = train_models(df)

    print("✅ Training completed")
