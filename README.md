# EPL Match Outcome Prediction

## Objective

The objective of this project is to predict the outcome of English Premier League (EPL) football matches  
(**Home win / Draw / Away win**) using historical match statistics and machine learning techniques.

This project is **academic in nature**.  
The goal is **not** to beat bookmakers, but to demonstrate a complete and correct **data science workflow**, including:
- data preparation
- feature engineering
- model training
- evaluation
- comparison with a strong baseline (bookmaker probabilities)

---

## Data

The dataset consists of multiple EPL seasons with match-level statistics.

Each match is represented by:
- historical team performance statistics
- rolling averages over recent matches
- differences between home and away teams

Only **pre-match information** is used to avoid data leakage.

The target variable is encoded as:
- '1' → Home win  
- '0'  → Draw  
- '-1' → Away win  

---

## Feature Engineering

The core feature engineering strategy is based on **home vs away differences**.

Features include:
- differences in recent points
- differences in goals scored and conceded
- differences in expected goals (xG)
- differences in shots, possession, discipline metrics, etc.

Rolling averages (e.g. last 5 or 10 matches) are used to represent **recent team form**.

The final list of features used for training is saved in:
models/features.txt


---

## Methodology

### Models

Two machine learning models are trained:

- **Logistic Regression**
  - baseline and interpretable model
  - trained using a standardized feature pipeline

- **Random Forest**
  - non-linear ensemble model
  - captures complex interactions between features

### Train-Test Split

A **temporal split** is used:
- first 80% of matches → training set  
- last 20% of matches → test set  

This ensures that future matches are never used to predict past outcomes.

---

## Evaluation

Model performance is evaluated using:
- Accuracy
- Log-loss
- Confusion matrices
- Full classification reports

Classification reports are saved as text files:
results/logistic_regression_report.txt
results/random_forest_report.txt


---

## Bookmaker Baseline

A bookmaker baseline is used as a reference.

Bookmaker implied probabilities are computed from match odds and normalized.
This baseline represents a strong benchmark, as bookmakers aggregate a large
amount of information not directly available in the dataset.

---

## Probabilistic Evaluation

Beyond accuracy, predicted probabilities are compared with bookmaker probabilities.

A model is considered to outperform the bookmaker on a given match if it assigns
a higher probability to the **true outcome** than the bookmaker does.

The probabilistic comparison results are saved to:
results/match_probabilities_comparison.csv


---

## Results Summary

- Both machine learning models achieve approximately **53–54% accuracy**
- Performance is comparable to the bookmaker baseline
- Draws remain difficult to predict, a limitation also observed for bookmakers
- In around **38% of matches**, the model assigns a higher probability to the
  true outcome than the bookmaker

These results are realistic and consistent with the difficulty of football
match outcome prediction.

---

## Reproducibility

To ensure reproducibility:
- Trained models are saved in `models/`
- The feature list used for training is saved in `models/features.txt`
- Evaluation reports and probabilistic comparisons are saved in `results/`
- The full software environment is specified in `environment.yml`

---

## Project Structure

├── main.py
├── src/
│ ├── data_loader.py
│ ├── models.py
│ ├── bookmaker_baseline.py
│ └── probabilistic_evaluation.py
├── data/
│ ├── raw/
│ └── processed/
├── models/
│ ├── logistic_regression.pkl
│ ├── random_forest.pkl
│ └── features.txt
├── results/
│ ├── logistic_regression_report.txt
│ ├── random_forest_report.txt
│ ├── random_forest_feature_importance.txt
│ └── match_probabilities_comparison.csv
├── notebooks/
│ └── 01_exploration_and_features.ipynb
├── tests/
│ └── test_pipeline.py
├── environment.yml
└── README.md


---

## Environment Setup -- with conda

Create the Conda environment:

conda env create -f environment.yml

Activate the environment:
conda activate epl-match-prediction-1

## Environment Setup -- without conda

Although using Conda is recommended—as it automatically manages both the Python version and package dependencies—the project does not rely on advanced environment-specific features.

This project was developed using Python 3.10. Please ensure that your environment includes the following dependencies:

  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
  - pytest
  - jupyter

## How to Run

Run the full pipeline:
python main.py

## Tests

Run the test suite:
python -m pytest


## Notes

This project follows good machine learning and software engineering practices:
- no data leakage
- temporal validation
- clear separation of responsibilities
- reproducible results
- transparent evaluation

It is intended as an academic demonstration of applied data science.


