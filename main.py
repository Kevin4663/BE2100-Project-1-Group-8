from StockAnalysis import StockAnalysis
from datetime import datetime

def main():
    tickers=["NVDA", "AMD", "MSFT", "AAPL"]
    sa1 = StockAnalysis(tickers, 
        start_date="2023-01-01", 
        end_date=datetime.today().strftime('%Y-%m-%d'),
        interval="1d", 
        path="stock_data.csv")
    sa1.fetch_data()
    for i in range(len(tickers)):
        sa1.plot_dashboard(i)

if __name__ == "__main__":
    main()