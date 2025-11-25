import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

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

    # Method to fetch stock data for a given ticker and range
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
    def plot_data(self):
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return
        
        
        for ticker in self.tickers:
            try:
                self.data[ticker]['Close'].plot(label=ticker)
            except KeyError:
                print(f"Could not find data for {ticker}")

        plt.title("Close Prices Comparison")
        plt.legend()
        plt.show()

    # Method to perform statistical analysis, statistical intervals and hypothesis test Part B and C

    # Method for ML algo Part D


# Test Example
if __name__ == "__main__":
    # Example: Fetch a set of stock data from Jan 1, 2024 to today
    sa1 = StockAnalysis(tickers=["NVDA", "AMD", "INTC"],
        start_date="2024-01-01",
        end_date=datetime.today().strftime('%Y-%m-%d'),
        interval="1d",
        path="stock_data.csv"
    )

    sa1.fetch_data()
    sa1.plot_data()