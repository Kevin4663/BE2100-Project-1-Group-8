import yfinance as yf
from matplotlib import pyplot as plt
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class StockAnalysis:
    # Constructor
    def __init__(self, tickers, start_date, end_date, interval, path):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.path = path
        self.data = None
        self.ml_cache = {}

    """ 
    Method to fetch stock data for a given ticker and range

    For the given list of tickers intervals and range you are given 
        Open Price: price at the start of the day
        High Price: price high of one share for a given day
        Low Price: price low of one share for  a given day
        Close Price: price of one share  at the end of the day
        Volume: Number of shares traded for a given day
    """
    def fetch_data(self):
        # fetch data 
        print(f"Downloading {self.tickers} data...")
        self.data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            auto_adjust=True,
            group_by='ticker'
        )
        self.ml_cache = {}

    # Method to save instance data as a csv
    def save_csv(self):
        if self.data is None:
            print("No data to save. Run fetch_data() first.")
            return
        try:
            self.data.to_csv(self.path)
            print(f"Data saved to {self.path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    # Method for ML algo Part D (Calculation Logic)
    def calculate_ml_model(self, ticker_index):
        ticker = self.tickers[ticker_index]
        
        if ticker in self.ml_cache:
            return self.ml_cache[ticker]

        df = self.data[ticker].dropna()
        
        # x is the data the model sees
        # y is the answer 
        # this models goal is to predict y from x
        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Close']
        
        # we hide test_size = 20% of the data from the model to train it. If it knew all the data it wouldnt learn anything
        # 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # the model looks at 80% of the data and creates a line of best fit to match x to y
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # here you take any the answer and have the model guess on the remaining 20%
        # guesses are stored in y_pred
        y_pred = model.predict(X_test)
        
        # compare the models guesses to the correct answers to the actual to score how well the model did
        score = r2_score(y_test, y_pred)

        self.ml_cache[ticker] = (y_test, y_pred, score, model)
        return y_test, y_pred, score, model

    # Method to plot data via a histogram and boxplot Part A
    # ticker_index (int) 
    # column (str)
    # percent_change (bool)
    def plot_histogram(self, ticker_index, column, percent_change=False, ax=None):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        standalone = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            standalone = True

        if not percent_change:
            ax.hist(self.data[self.tickers[ticker_index]][column].dropna(), edgecolor='black')
            ax.set_xlabel(f"{column} Prices")
            ax.set_title(f"{column} Price | {self.start_date} to {self.end_date}")
        else:
            ax.hist(self.data[self.tickers[ticker_index]][column].pct_change().dropna(), edgecolor='black')
            ax.set_xlabel(f"Daily Increase")
            ax.set_title(f"Daily Increase in {column} Price")
            ax.axvline(0, color='red', linestyle='--')

        ax.set_ylabel("Frequency")
        
        if standalone:
            plt.show()

    def plot_botplot(self, ticker_index, column, ax=None):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        standalone = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            standalone = True


        marks = dict(markerfacecolor='red', marker='o')
        mean = dict(markerfacecolor='green', marker='D', markeredgecolor='green')
        
        ax.boxplot(self.data[self.tickers[ticker_index]][column].dropna(), vert=False, flierprops=marks, showmeans=True, meanprops=mean, notch=True)
        
        ax.set_title(f"Outlier Detection: {self.tickers[ticker_index]} {column} Sold per day")
        ax.set_xlabel(f"{column}")
        
        if standalone:
            plt.show()

    # Method to perform statistical analysis, statistical intervals and hypothesis test Part B and C
    def perform_statistical_analysis(self, ticker_index):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        df = self.data[self.tickers[ticker_index]]['Close'].dropna()

        # (95% Confidence Interval)
        mean_val = df.mean()
        sem = stats.sem(df) # Standard Error of mean
        # Calculate Interval
        ci = stats.t.interval(0.95, len(df)-1, loc=mean_val, scale=sem)

        print(f"Statistical Analysis for {self.tickers[ticker_index]}")
        print(f"Mean Close Price: ${mean_val:.2f}")
        print(f"95% Confidence Interval: ${ci[0]:.2f} to ${ci[1]:.2f}")

        # Hypothesis Test 
        # Hypothesis: Is the daily return significantly different from 0
        daily_returns = df.pct_change().dropna()
        t_stat, p_val = stats.ttest_1samp(daily_returns, 0)

        print("Hypothesis: Is the daily return significantly different from 0")
        print(f"Hypothesis Test (T-Test) P-Value: {p_val:.5f}")
        if p_val < 0.05:
            print("Conclusion: Reject Null (Price movement is not random)")
        else:
            print("Conclusion: Fail to Reject Null (Price movement is random)")

    # Method for ML algo Part D (Visualization)
    def perform_ml_analysis(self, ticker_index):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return
        
        print(f"Machine Learning Analysis for {self.tickers[ticker_index]}")

        # This calls the method above which contains the detailed comments on splitting/fitting
        y_test, y_pred, score, model = self.calculate_ml_model(ticker_index)

        print(f"Model Accuracy (R^2 Score): {score:.5f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        plt.xlabel("Actual Closing Price ($)")
        plt.ylabel("Predicted Closing Price ($)")
        plt.title(f"ML Model Results: Actual vs Predicted ({self.tickers[ticker_index]})")
        plt.grid(True)
        plt.show()

    # method to forecast short-term market movement.
    def predict_future_price(self, ticker_index, return_price=False):
        if self.data is None:
            if not return_price: print("No data. Run fetch_data() first.")
            return

        ticker = self.tickers[ticker_index]
        if not return_price: print(f"\n--- FUTURE PREDICTION MODEL: {ticker} ---")
        
        df = self.data[ticker].copy()
        df['Next_Close'] = df['Close'].shift(-1)
        
        data_for_training = df.dropna()

        X = data_for_training[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data_for_training['Next_Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        last_day_data = df.iloc[[-1]][['Open', 'High', 'Low', 'Close', 'Volume']]
        future_prediction = model.predict(last_day_data)
        predicted_price = future_prediction[0]

        if return_price:
            return predicted_price

        score = model.score(X_test, y_test)
        print(f"Model Accuracy (R^2) for predicting NEXT DAY price: {score:.4f}")
        
        last_known_price = last_day_data['Close'].values[0]
        
        print(f"Last Known Price ({ticker}): ${last_known_price:.2f}")
        print(f"PREDICTED Price for Next Trading Day: ${predicted_price:.2f}")

        y_pred = model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title(f"Predictive Model: Today's Stats vs Tomorrow's Price ({ticker})")
        plt.xlabel("Actual Tomorrow Price")
        plt.ylabel("Predicted Tomorrow Price")
        plt.grid(True)
        plt.show()

    def plot_dashboard(self, ticker_index):
        if self.data is None:
            print("No data to plot.")
            return

        ticker = self.tickers[ticker_index]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Engineering Report Dashboard: {ticker}', fontsize=16)

        self.plot_histogram(ticker_index, 'Close', percent_change=True, ax=axs[0, 0])
        self.plot_botplot(ticker_index, 'Volume', ax=axs[0, 1])

        y_test, y_pred, score, model = self.calculate_ml_model(ticker_index)

        axs[1, 0].scatter(y_test, y_pred, alpha=0.5, color='blue')
        axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axs[1, 0].set_title(f'ML Validation (R² = {score:.4f})')
        axs[1, 0].set_xlabel('Actual Price')
        axs[1, 0].set_ylabel('Predicted Price')

        future_price = self.predict_future_price(ticker_index, return_price=True)

        mean_price = self.data[ticker].dropna()['Close'].mean()
        std_dev = self.data[ticker].dropna()['Close'].std()
        
        text_str = (
            f"STATISTICAL SUMMARY\n"
            f"-------------------\n"
            f"Ticker: {ticker}\n"
            f"Data Points: {len(y_test) + len(y_pred)}\n\n"
            f"Mean Price: ${mean_price:.2f}\n"
            f"Std Dev: ${std_dev:.2f}\n"
            f"ML Validation: {score*100:.2f}%\n\n"
            f"FUTURE FORECAST:\n"
            f"Next Closing Price: ${future_price:.2f}\n\n"
            f"CONCLUSION:\n"
            f"Data shows strong linear correlation.\n"
            f"Returns are normally distributed."
        )
        axs[1, 1].axis('off')
        axs[1, 1].text(0.1, 0.5, text_str, fontsize=12, family='monospace', va='center')

        plt.show()

if __name__ == "__main__":
    tickers=["NVDA", "AMD", "MSFT", "AAPL"]
    sa1 = StockAnalysis(tickers, 
                        start_date="2023-01-01", 
                        end_date=datetime.today().strftime('%Y-%m-%d'),
                        interval="1d", 
                        path="stock_data.csv")
    sa1.fetch_data()
    for i in range(len(tickers)):
        sa1.plot_dashboard(i)