from StockAnalysis import StockAnalysis

def main():
    sa1 = StockAnalysis("NVDA", "2024-01-01", "2025-01-01", "1d", path = None)
    sa1.fetch_data()
    sa1.plot_data()


if __name__ == "__main__":
    main()



