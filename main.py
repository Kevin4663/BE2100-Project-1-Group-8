from fetch_csv import fetch_csv

def main():
    ticker = "AAPL"

    fetch_csv(ticker)
    

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
from datetime import datetime

# fetch a csv for a ticker from start to end in set intervals saved to a given pth
def fetch_csv(company_ticker):
    ticker = yf.Ticker(company_ticker)
    print(ticker.financials)


