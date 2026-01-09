**Project Title: Predicting Football Match Outcomes Using Real Match Data**

**Category** : Data Science / Sports Analytics

Football is a complex sport influenced by both skill and situational factors. While luck plays a role, team performance often follows identifiable statistical patterns such as shots on target, possession, and goal differential. The goal of this project is to use real-world football data to understand and predict match results. Specifically, I will analyze historical Premier League match statistics to predict whether the home team wins, draws, or loses.

This project aims to answer the question: *Can we predict the outcome of a football match using in-game statistics?*. 
By doing so, it demonstrates how data science can uncover the relationships between performance metrics and success in sports.

The planned approach approximately follows these steps:
1) Data collection : use publicly available datasets
2) Data cleaning and preparation : convert results into numerical labels, remove incomplete rows and normalize numeric values, create derived metrics (difference in shots, possesion gaps etc)
3) Data analysis : identify which factors correlate most strongly with winning, visualize distributions using matplotlib and seaborn
4) Model building : use real match data to train a predictive model that will estimate the probability of win, draw, loss based on match statistics
5) Model evaluation : split data into training and testing sets, evaluate prediction accuracy and measure which features contribute most to accurate predictions
6) Visualisation and interpretation : show actual vs predicted outcomes, rank the most important features

I plan to use python 3.10+, pandas, NumPy, matplotlib and seaborn for this project, maybe something else if needed.

The expected challenges are : 
- Data quality and consistency : some seasons may have missing or inconsistent values or naming, so I’ll clean and standardize the data carefully and focus on complete seasons
- Feature selection, choosing the right variables : not all match statistics are equally important for predicting results

The project will be considered successful if it: uses real football data, achieves reasonable prediction accuracy on unseen matches, identifies and explains which statistics are most predictive of results and provides clear, interpretable visualizations connecting data science and football performance.

If time permits, extend the model to predict exact score instead of just win/draw/loss and potentially compare results across different league

Reminder : This was my proposal before going in the project. the core of the work remain the same but has evolved doing it.