from pathlib import Path
import pandas as pd

from src.data_loader import (
    load_raw,
    load_base,
    prepare_home_away,
    pivot_matches,
    build_all,
    merge_dataset,
    build_data_before_engineering,
    build_team_rolling_features,
    build_match_level_features,
)
from src.models import evaluate_bookmaker, train_models

from src.probabilistic_evaluation import run_probabilistic_evaluation



def main():
    print("""
    =================================================
    - EPL MATCH OUTCOME PREDICTION -
    -------------------------------------------------
    This script:
    1. Builds a clean match-level dataset
    2. Engineers rolling team features
    3. Trains ML models (LogReg, RF)
    4. Compares predictions to bookmaker odds
    =================================================
    """)


    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)


    print("▶ Step 1: build matchdata_base.csv")
    df_base = load_raw()
    df_base.to_csv("data/processed/matchdata_base.csv", index=False)
    print("matchdata_base.csv done")


    print("▶ Step 2: build matchdata_clean.csv")
    df_loaded = load_base()
    home, away = prepare_home_away(df_loaded)
    df_clean = pivot_matches(home, away)
    df_clean.to_csv("data/processed/matchdata_clean.csv", index=False)
    print("matchdata_clean.csv done")

 
    print("▶ Step 3: build all_matches_clean.csv")
    df_all = build_all()
    df_all.to_csv("data/processed/all_matches_clean.csv", index=False)
    print("all_matches_clean.csv done")


    print("▶ Step 4: merge datasets")
    df_merged = merge_dataset(df_clean, df_all)
    df_merged.to_csv("data/processed/data_merged.csv", index=False)
    print("data_merged.csv done")


    print("▶ Step 5: team-level table")
    df_before = build_data_before_engineering(df_merged)

    df_before.to_csv(
        "data/processed/data_before_engineering.csv",
        index=False
    )
    print("data_before_engineering.csv done")



    print("▶ Step 6: rolling features")
    df_after = build_team_rolling_features(df_before)

    df_after.to_csv(
        "data/processed/data_after_engineering.csv",
        index=False
    )
    print("data_after_engineering.csv done")



    print("▶ Step 7: final ML dataset")
    df_model = build_match_level_features(df_after)
    df_model.to_csv("data/processed/model_data.csv", index=False)
    print("model_data.csv done")

    print("\n▶ Step 8: bookmaker baseline evaluation")
    split_idx = int(len(df_model) * 0.8)
    df_test = df_model.iloc[split_idx:].reset_index(drop=True)
    book_metrics = evaluate_bookmaker(df_test)

    
    print("\n▶ Step 9: training ML models")

    df_model["match_date"] = pd.to_datetime(df_model["match_date"])
    df_model = df_model.sort_values("match_date").reset_index(drop=True)
    df_model = df_model.dropna().reset_index(drop=True)

    log_model, rf_model, metrics = train_models(df_model, book_metrics)

    Path("results").mkdir(exist_ok=True)

    split_idx = int(len(df_model) * 0.8)
    df_test = df_model.iloc[split_idx:].reset_index(drop=True)


    print("\n▶ Step 10: probabilistic model vs bookmaker evaluation")
    run_probabilistic_evaluation()


    print("PIPELINE FINISHED SUCCESSFULLY")




if __name__ == "__main__":
    main()




