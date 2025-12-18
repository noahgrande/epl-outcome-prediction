# Abstract

Predicting football match outcomes is a challenging task due to the complex and dynamic nature of the sport, where team performance, tactical choices, and situational factors all play an important role. This project investigates whether historical match statistics can be used to predict the outcome of football matches, defined as a home win, draw, or away win.

Using real Premier League data, a dataset was constructed from publicly available match statistics and bookmaker odds from season 2021/22 to 2024/25 representing approximately 1400 matches. The data preparation process included cleaning, merging, feature selection, and the creation of rolling performance indicators capturing differences between the home and away teams. Several machine learning models were implemented, including logistic regression and random forest classifiers, to estimate outcome probabilities.

Model performance was evaluated using standard classification metrics such as accuracy, log loss, and confusion matrices on unseen test data. In addition, model predictions were compared to bookmaker implied probabilities, which serve as a strong baseline reflecting aggregated market information. The results show that the models achieve reasonable predictive performance and capture meaningful relationships between match statistics and outcomes, while still exhibiting limitations when compared to bookmaker probabilities.

This project oulines a full data science workflow, from data preparation to model evaluation, and highlights both the potential and the constraints of using statistical learning methods for predicting sports outcome.


# 1. Introduction

Football is one of the most widely followed sports in the world. But it is more than just a global passion, it is also a massive generator of publicly available data related to team performance, match events, and outcomes. This load of data has made football an attractive domain for data science and machine learning applications, particularly for the analysis and prediction of match results. Yet, despite this digital goldmine, the beautiful game remains extremely challenging to forecast. The outcome of a match is often hard to predict due to the influence of uncertainty, randomness, and contextual factors such as home advantage, team form, and tactical decisions.

The objective of this project is to investigate whether historical match statistics can be used to predict the outcome of football matches, defined as a home win, draw, or away win. Instead of trying to create a betting system or outperform bookmakers, the project adopts an academic perspective focused on understanding the relationship between the performance indicators and the result. In this context, bookmakers odds are not viewed as a target to beat but as a sophisticated benchmark representing the intelligence of the market to be compared in the end with the model.

To address this problem, I constructed a dataset using real match statistics and betting odds. The features were designed to capture relative differences between the home and away teams, including recent performance, goal-related metrics, and advanced statistics such as expected goals. I used two machine learning models including logistic regression and random forest classifiers, in order to estimate outcome probabilities based on these features.

This report walks trough the entire process, from feature engineering and model training to a critical look at where statistical models succeed and where they fall short.



# 2. Literature Review

Football match outcome prediction has become an active area of research with the growing availability of detailed match statistics and advances in machine learning techniques. In most studies, the task is formulated as a classification problem, where match outcomes are predicted as a home win, draw, or away win. Despite the increasing sophistication of predictive models, accurately forecasting football results remains difficult due to the inherent uncertainty of the sport and the imbalanced distribution of match outcomes.

Early research in this domain primarily relied on statistical models such as Poisson regression to model goal-scoring processes. These approaches provided interpretable probabilistic frameworks but were limited in their ability to capture complex relationships between teams and contextual factors. As a result, more recent work has shifted toward machine learning methods, including logistic regression, support vector machines, random forests, and gradient boosting models, which are better suited to handling non-linear patterns and higher-dimensional feature spaces.

A recurring challenge highlighted in the literature is the difficulty of predicting draws in multiclass classification settings. Several studies report that machine learning models tend to perform well in distinguishing wins from losses, while draws are often poorly predicted or entirely ignored by the models. This issue is commonly attributed to class imbalance, as draws occur less frequently than wins or losses. To address this problem, some researchers reformulate the prediction task as a binary classification problem, such as predicting home win versus non-home win, which has been shown to improve overall predictive performance.

Feature engineering plays a central role in football match prediction. Many studies emphasize the importance of using features that are known prior to match kickoff in order to ensure realistic predictive scenarios. Commonly used features include aggregated match statistics from previous games, such as shots, goals, possession, and disciplinary records, often computed as averages over a fixed number of recent matches. In addition, advanced performance indicators such as expected goals (xG) have gained popularity, as they provide a more informative representation of chance quality than raw goal counts.

Another widely adopted strategy is the use of relative features that capture the difference in performance or strength between the home and away teams. By focusing on differences rather than absolute values, these features directly encode competitive balance and home advantage effects, which are known to influence match outcomes. Ratings derived from external sources, such as FIFA ratings or league rankings, are also frequently used as proxies for team strength and long-term performance.

Finally, bookmaker odds are often used in the literature as a benchmark for evaluating predictive models. Bookmakers aggregate large amounts of information and market expectations, making their implied probabilities difficult to outperform consistently. Rather than treating betting profitability as the primary objective, several studies use bookmaker odds as a baseline to assess whether statistical or machine learning models capture meaningful information beyond what is already reflected in market prices.

Overall, existing research suggests that machine learning models can achieve reasonable predictive performance when applied to football match data, particularly when careful feature engineering and problem formulation are employed. However, limitations related to class imbalance, draw prediction, and model generalization remain significant. This project builds on prior work by adopting a structured machine learning approach, emphasizing relative performance features, probabilistic evaluation, and comparison to bookmaker-based baselines within an academic data science framework.



