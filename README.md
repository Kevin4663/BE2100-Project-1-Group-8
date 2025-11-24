# BE 2100 Project: Stock Market Trend Prediction (AAPL)

## Project Overview
This project fulfills the requirements for the **BE 2100 Final Project (Option 1: Engineering Problem)**. 

The objective is to apply engineering statistical analysis and machine learning techniques to historical financial data. We utilize Python to analyze the daily price movements of Apple Inc. (AAPL), investigating whether intra-day metrics (Open, High, Low, Volume) are statistically significant predictors of the final Closing price.

## Features
This software performs the complete statistical workflow required by the project rubric:
1.  **Data Collection:** Automatically fetches 2 years of historical data using the Yahoo Finance API (`yfinance`).
2.  **Descriptive Statistics:** Generates frequency distributions (Histograms) and outlier detection (Box Plots) for daily returns and trading volume.
3.  **Statistical Intervals:** Calculates the 95% Confidence Interval for the population mean of the stock price.
4.  **Hypothesis Testing:** Performs a One-Sample T-test to determine if daily returns are statistically non-zero.
5.  **Machine Learning:** Implements a Linear Regression model (`scikit-learn`) to predict Closing prices based on intra-day features, calculating $R^2$ accuracy and Mean Squared Error.

## Prerequisites
Ensure you have **Python 3.8+** installed. You will need the following libraries:

* `yfinance` (Data fetching)
* `pandas` (Data manipulation)
* `matplotlib` & `seaborn` (Data visualization)
* `scipy` (Statistical calculations)
* `scikit-learn` (Machine Learning algorithms)

## Installation
1.  Clone this repository or extract the project files.
2.  Open your terminal/command prompt to the project directory.
3.  Install the required dependencies using `pip`:

```bash
pip install yfinance pandas matplotlib seaborn scikit-learn scipy
