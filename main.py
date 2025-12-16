from pathlib import Path

# import des fonctions depuis src/
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

def main():
    print("🚀 Starting project pipeline")

    # ----------------------------------
    # 1. matchdata_base.csv
    # ----------------------------------
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("▶ Step 1: build matchdata_base.csv")
    df_base = load_raw()
    df_base.to_csv("data/processed/matchdata_base.csv", index=False)
    print("✅ matchdata_base.csv done")

    # ----------------------------------
    # 2. matchdata_clean.csv
    # ----------------------------------
    print("▶ Step 2: build matchdata_clean.csv")
    df_loaded = load_base()
    home, away = prepare_home_away(df_loaded)
    df_clean = pivot_matches(home, away)
    df_clean.to_csv("data/processed/matchdata_clean.csv", index=False)
    print("✅ matchdata_clean.csv done")

    # ----------------------------------
    # 3. all_matches_clean.csv
    # ----------------------------------
    print("▶ Step 3: build all_matches_clean.csv")
    df_all = build_all()
    df_all.to_csv("data/processed/all_matches_clean.csv", index=False)
    print("✅ all_matches_clean.csv done")

    # ----------------------------------
    # 4. merge datasets
    # ----------------------------------
    print("▶ Step 4: merge datasets")
    df_merged = merge_dataset(df_clean, df_all)
    df_merged.to_csv("data/processed/data_merged.csv", index=False)
    print("✅ data_merged.csv done")

    # ----------------------------------
    # 5. before feature engineering
    # ----------------------------------
    print("▶ Step 5: team-level table")
    df_before = build_data_before_engineering(df_merged)

    # ----------------------------------
    # 6. rolling features
    # ----------------------------------
    print("▶ Step 6: rolling features")
    df_after = build_team_rolling_features(df_before)

    # ----------------------------------
    # 7. match-level ML dataset
    # ----------------------------------
    print("▶ Step 7: final ML dataset")
    df_model = build_match_level_features(df_after)
    df_model.to_csv("data/processed/model_data.csv", index=False)
    print("✅ model_data.csv done")

    print("🎉 PIPELINE FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()



