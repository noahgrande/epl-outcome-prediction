from pathlib import Path
import pandas as pd
import joblib


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_models(df: pd.DataFrame):

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

    print("Logistic regression summary saved to results/logistic_regression_coefficient.txt")

    y_pred_log = log_reg.predict(x_test)
    y_proba_log = log_reg.predict_proba(x_test)

    log_accuracy = accuracy_score(y_test, y_pred_log)
    log_cm = confusion_matrix(y_test, y_pred_log)
    log_ll = log_loss(
        y_test,
        y_proba_log,
        labels=log_reg.named_steps["clf"].classes_
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



    return log_reg, rf, metrics
    
