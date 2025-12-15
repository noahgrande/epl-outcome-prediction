"""
Prediction script for EPL outcome prediction model.

Allows:
- Predicting a match by entering home & away team names
- Uses most recent PRE-MATCH feature snapshot
- Outputs calibrated probabilities
"""

import pandas as pd
import pickle


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
def load_dataset(path="data/processed/data_merged_with_features.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


# ---------------------------------------------------------
# Load selected feature list
# ---------------------------------------------------------
def load_feature_list(path="models/feature_list.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


# ---------------------------------------------------------
# Decode prediction
# ---------------------------------------------------------
def decode_prediction(label):
    return {-1: "Away Win", 0: "Draw", 1: "Home Win"}[label]


# ---------------------------------------------------------
# Predict by entering team names
# ---------------------------------------------------------
def predict_match_by_teams(df, model, features):

    print("\nEnter the match you want to predict:")
    home = input("Home team: ").strip().lower()
    away = input("Away team: ").strip().lower()

    # Filter historical matches for this pairing
    subset = df[
        (df["home_team"].str.lower() == home) &
        (df["away_team"].str.lower() == away)
    ]

    if subset.empty:
        print("\n❌ No historical match found for this pairing.")
        print("The model requires historical data to build pre-match features.")
        return

    # -----------------------------------------------------
    # IMPORTANT:
    # We take the MOST RECENT match as a proxy
    # for PRE-MATCH information of a future encounter
    # -----------------------------------------------------
    row = subset.sort_values("match_date").iloc[-1]

    X = row[features].to_frame().T

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    proba_dict = dict(zip(classes, proba))
    pred_class = max(proba_dict, key=proba_dict.get)

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------
    print("\n==============================")
    print("        🔮 MODEL PREDICTION")
    print("==============================")
    print(f"Match: {row['home_team']} vs {row['away_team']}")
    print(f"Prediction: {decode_prediction(pred_class)}")

    print("\n📊 Probabilities:")
    print(f"   Home Win: {proba_dict.get(1, 0):.3f}")
    print(f"   Draw:     {proba_dict.get(0, 0):.3f}")
    print(f"   Away Win: {proba_dict.get(-1, 0):.3f}")
    print("==============================\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    print("\n==============================")
    print("   ⚽ EPL OUTCOME PREDICTOR")
    print("==============================\n")

    model = load_model("models/logistic_regression_model.pkl")
    df = load_dataset()
    features = load_feature_list()

    print(f"✔ Loaded model and {len(features)} features.")
    print("Ready to predict.\n")

    predict_match_by_teams(df, model, features)


