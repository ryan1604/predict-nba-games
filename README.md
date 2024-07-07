# Predict NBA Games with Machine Learning

This project consists of two parts:

- Getting data on NBA games by web scraping [basketball-reference.com](http://basketball-reference.com)
- Using machine learning to predict outcomes of NBA games

## Overview

**Part 1:**

Using `Playwright`, data from each NBA season is collected, and information from each game is stored locally. 
Afterward, the data from each game is parsed and key information such as total score for each team, basic stats, 
and advanced stats are stored in a `CSV` file to be used for machine learning. The tables in the data are cleaned using `BeautifulSoup4`.

**Part 2:**

From the `CSV` file, using `pandas` and `scikit-learn`, the outcomes of NBA games are predicted from its historical data. 
It demonstrates data preprocessing, feature selection, and machine learning to predict the outcomes.

**Some of the steps are as follows:**

1. Data cleaning and preparation by removing irrelevant columns and handling missing values.
2. Generation of new target variables based on future game results.
3. Time series cross-validation to avoid lookahead bias in model training.
4. Normalization of features using `MinMaxScaler`.
5. Feature selection using `SequentialFeatureSelector` with `RidgeClassifier`.
6. Backtesting to simulate the model's performance on unseen data.
7. Calculation of rolling averages for time series data to capture trends.
8. Adding new features, such as whether a team will play at home next game.
9. Finally, merge all features and select the best features using `SequentialFeatureSelector` again.

`MinMaxScaler` is used to prevent bias in models where feature scales significantly differ, 
which can disproportionately influence the model's performance.

## Updates

Backward elimination improves model by `1.1%` compared to forward selection (previously `62.3%`).

## Further Improvements

The model can be further improved by:

- Using `RandomForest` instead of `ridge regression`.
- Adjusting the number of features selected.
- Using backward elimination instead of forward selection.
- Include win-loss ratio as a feature.
