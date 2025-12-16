import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_models(df: pd.DataFrame):
    """
    Entraîne une Logistic Regression et un Random Forest
    avec un split temporel (80 / 20).

    Retourne:
        (log_reg_model, rf_model, metrics_dict)
    """

    # --------------------------------------------------
    # 0) Sécurité: tri temporel + dropna (évite fuite et erreurs)
    # --------------------------------------------------
    df = df.copy()
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        df = df.sort_values("match_date").reset_index(drop=True)

    # On garde uniquement les lignes où features et target sont présents
    X = df.filter(regex="^diff_")
    y = df["target"]
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # --------------------------------------------------
    # 1) Split temporel 80 / 20
    # --------------------------------------------------
    split_idx = int(len(df) * 0.8)
    # IMPORTANT: split sur X/y filtrés (valid) -> on split sur leur longueur
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    metrics = {}

    # --------------------------------------------------
    # 2) Logistic Regression (AVEC scaling via Pipeline)
    # --------------------------------------------------
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        ))
    ])

    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    y_proba_log = log_reg.predict_proba(X_test)

    print("\n📊 Logistic Regression")
    print("Accuracy:", accuracy_score(y_test, y_pred_log))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_log))
    print(classification_report(y_test, y_pred_log, zero_division=0))

    # Log-loss = métrique probabiliste (utile vs bookmakers)
    metrics["log_reg"] = {
        "accuracy": accuracy_score(y_test, y_pred_log),
        "log_loss": log_loss(y_test, y_proba_log, labels=log_reg.named_steps["clf"].classes_),
        "classes": list(log_reg.named_steps["clf"].classes_),
    }

    # --------------------------------------------------
    # 3) Random Forest (pas besoin de scaling)
    # --------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)

    print("\n🌲 Random Forest")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    metrics["rf"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "log_loss": log_loss(y_test, y_proba_rf, labels=rf.classes_),
        "classes": list(rf.classes_),
    }

    return log_reg, rf, metrics
    
