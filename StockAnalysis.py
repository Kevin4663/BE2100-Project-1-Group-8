import yfinance as yf
from matplotlib import pyplot as plt
from datetime import datetime
from scipy import stats


class StockAnalysis:
    # Constructor
    def __init__(self, tickers, start_date, end_date, interval, path):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.path = path

        self.data = None

    # Implement the following methods

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
    
    # Method to plot data via a histogram and boxplot Part A
    # ticker_index (int) 
    # column (str)
    # percent_change (bool)

    def plot_histogram(self, ticker_index, column, percent_change = False):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return
        plt.figure(figsize=(10,6))

        if not percent_change:
            plt.hist(self.data[self.tickers[ticker_index]][column].dropna(), edgecolor='black')
            plt.xlabel(f"{column} Prices")
            plt.title(f"{column} Price | {self.start_date} to {self.end_date} n={self.data.shape[0]}")
        else:
            plt.hist(self.data[self.tickers[ticker_index]][column].pct_change().dropna(), edgecolor='black')
            plt.xlabel(f"Daily Increase")
            plt.title(f"Daily Increase in {column} Price | {self.start_date} to {self.end_date} n={self.data.shape[0]}")

        plt.ylabel("Frequency")
        plt.show()

    def plot_botplot(self, ticker_index, column):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        plt.figure(figsize=(10, 6))

        marks = dict(markerfacecolor='red', marker='o')
        mean = dict(markerfacecolor='green', marker='D', markeredgecolor='green')
        plt.boxplot(self.data[self.tickers[ticker_index]][column].dropna(), vert=False, flierprops=marks, showmeans=True, meanprops=mean, notch=True)
        
        plt.title(f"Outlier Detection: {self.tickers[ticker_index]} {column} Sold per day")
        plt.xlabel(f"{column}")
        plt.show()

    # Method to perform statistical analysis, statistical intervals and hypothesis test Part B and C
    def perform_statistical_analysis(self, ticker_index):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        series = self.data[self.tickers[ticker_index]]['Close'].dropna()

        # (95% Confidence Interval)
        mean_val = series.mean()
        sem = stats.sem(series) # Standard Error of mean
        # Calculate Interval
        ci = stats.t.interval(0.95, len(series)-1, loc=mean_val, scale=sem)

        print(f"Statistical Analysis for {self.tickers[ticker_index]}")
        print(f"Mean Close Price: ${mean_val:.2f}")
        print(f"95% Confidence Interval: ${ci[0]:.2f} to ${ci[1]:.2f}")

        # Hypothesis Test 
        # Hypothesis: Is the daily return significantly different from 0
        daily_returns = series.pct_change().dropna()
        t_stat, p_val = stats.ttest_1samp(daily_returns, 0)

        print("Hypothesis: Is the daily return significantly different from 0")
        print(f"Hypothesis Test (T-Test) P-Value: {p_val:.5f}")
        if p_val < 0.05:
            print("Conclusion: Reject Null (Price movement is not random)")
        else:
            print("Conclusion: Fail to Reject Null (Price movement is random)")
    
    # Method for ML algo Part D
    def perform_ml_analysis(self, ticker_index):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

# Test Example
if __name__ == "__main__":
    # Example: Fetch a set of stock data from Jan 1, 2024 to today
    sa1 = StockAnalysis(tickers=["AMD"],
        start_date="2025-01-01",
        end_date=datetime.today().strftime('%Y-%m-%d'),
        interval="1d",
        path="stock_data.csv"
    )

    sa1.fetch_data()
    #sa1.plot_histogram(0, 'Close', True)
    #sa1.plot_botplot(0, 'Volume')
    sa1.perform_statistical_analysis(0)
