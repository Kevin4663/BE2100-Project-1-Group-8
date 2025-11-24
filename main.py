from StockAnalysis import StockAnalysis

def main():
    sa1 = StockAnalysis("NVDA", "2024-01-01", "2025-01-01", "1d", "temp.csv")
    sa1.download_csv()

if __name__ == "__main__":
    main()



