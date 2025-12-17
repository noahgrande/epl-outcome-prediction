import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.bookmaker_baseline import evaluate_bookmaker
from src.probabilistic_evaluation import bookmaker_probabilities


# ======================================================
# FIXTURES
# ======================================================

def load_model_data():
    path = Path("data/processed/model_data.csv")
    assert path.exists(), "model_data.csv does not exist"
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


# ======================================================
# DATA INTEGRITY TESTS
# ======================================================

def test_dataset_not_empty():
    df = load_model_data()
    assert len(df) > 0


def test_required_columns_present():
    df = load_model_data()
    required_cols = ["match_id", "match_date", "target"]
    for col in required_cols:
        assert col in df.columns


def test_target_values_valid():
    df = load_model_data()
    assert set(df["target"].unique()).issubset({-1, 0, 1})


def test_no_nan_in_features():
    df = load_model_data()
    X = df.filter(regex="^diff_")
    assert not X.isna().any().any()


def test_match_id_unique():
    df = load_model_data()
    assert df["match_id"].is_unique


# ======================================================
# TEMPORAL CONSISTENCY TESTS
# ======================================================

def test_matches_sorted_by_date():
    df = load_model_data()
    assert df["match_date"].is_monotonic_increasing


def test_temporal_train_test_split():
    df = load_model_data()
    split_idx = int(len(df) * 0.8)

    train_max_date = df.iloc[:split_idx]["match_date"].max()
    test_min_date = df.iloc[split_idx:]["match_date"].min()

    # Allow same-day matches, forbid future leakage
    assert train_max_date <= test_min_date


# ======================================================
# FEATURE ENGINEERING TESTS
# ======================================================

def test_rolling_features_exist():
    df = load_model_data()
    rolling_cols = [c for c in df.columns if "_L5" in c or "_L10" in c]
    assert len(rolling_cols) > 0


def test_rolling_features_no_nan():
    df = load_model_data()
    rolling_cols = [c for c in df.columns if "_L5" in c or "_L10" in c]
    assert df[rolling_cols].notna().all().all()


# ======================================================
# MODEL TESTS
# ======================================================

def test_model_probabilities_valid():
    df = load_model_data()

    X = df.filter(regex="^diff_")
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    # Probabilities between 0 and 1
    assert ((proba >= 0) & (proba <= 1)).all()

    # Probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # Model predicts 3 classes
    assert set(model.classes_) == {-1, 0, 1}


def test_model_predicts_valid_classes():
    df = load_model_data()
    X = df.filter(regex="^diff_")
    y = df["target"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert set(preds).issubset({-1, 0, 1})


def test_logistic_coefficients_finite():
    df = load_model_data()
    X = df.filter(regex="^diff_")
    y = df["target"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    assert np.isfinite(model.coef_).all()


def test_model_reproducibility():
    df = load_model_data()
    X = df.filter(regex="^diff_")
    y = df["target"]

    model1 = LogisticRegression(random_state=42, max_iter=1000)
    model2 = LogisticRegression(random_state=42, max_iter=1000)

    model1.fit(X, y)
    model2.fit(X, y)

    assert np.allclose(model1.coef_, model2.coef_)


# ======================================================
# BOOKMAKER BASELINE TESTS
# ======================================================

def test_bookmaker_probabilities_consistent():
    df = load_model_data()

    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    metrics = evaluate_bookmaker(df_test)

    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["log_loss"] > 0


def test_bookmaker_probabilities_sum_to_one():
    df = load_model_data().iloc[:20]

    probs = df.apply(bookmaker_probabilities, axis=1, result_type="expand")
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ======================================================
# SANITY CHECK
# ======================================================

def test_accuracy_above_random_baseline():
    df = load_model_data()
    X = df.filter(regex="^diff_")
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # Better than random guessing (~33%)
    assert acc > 0.33




