import yfinance as yf
import pandas as pd
from datetime import datetime

# fetch a csv for a ticker from start to end in set intervals saved to a given pth
def fetch_csv(company_ticker):
    ticker = yf.Ticker(company_ticker)
    print(ticker.financials)

    data = yf.download(ticker, start='2023-01-01', end='2024-01-01')
    print(1)

