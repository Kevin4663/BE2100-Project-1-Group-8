import yfinance as yf
from datetime import datetime

class StockAnalysis:

    # constructor
    def __init__(self, company_ticker, start_date, end_date, interval, path):
        self.company_ticker = company_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.path = path
        self.data = None

    # Downloads stock data for a given ticker and range, saves it to a csv
    def download_csv(self):
        # download data 
        print(f"Downloading {self.company_ticker} data:")
        self.data = yf.download(self.company_ticker, start=self.start_date, end=self.end_date, interval=self.interval)

        # Validation
        if not self.data.empty:
            # Save to csv
            self.data.to_csv(self.path)
            print(f"Success! Data saved to {self.path}")
        else:
            print("An Error Occured, no data retrived")

    # Implement the following methods
    # Method to plot data via a histogram and boxplot Part A

    # Method to perform statistical analysis, statistical intervals and hypothesis test Part B and C

    # Method for ML algo Part D


# Test Example
if __name__ == "__main__":
    # Example: Fetch NVIDIA data from Jan 1, 2024 to today
    sa1 = StockAnalysis(company_ticker="NVDA",
        start_date="2024-01-01",
        end_date=datetime.today().strftime('%Y-%m-%d'),
        interval="1d",
        path="nvda_stock_data.csv")
    sa1.download_csv()