"""
BE 2100 Stock Analysis - Main Application
Enhanced version with comprehensive analysis features
"""

from StockAnalysis import StockAnalysis
from datetime import datetime
from typing import List

def run_basic_analysis(sa: StockAnalysis, tickers: List[str]) -> None:
    """Run basic dashboard analysis for all tickers."""
    print("\n" + "="*60)
    print("RUNNING BASIC DASHBOARD ANALYSIS")
    print("="*60 + "\n")
    
    for i in range(len(tickers)):
        print(f"\nGenerating dashboard for {tickers[i]}...")
        sa.plot_dashboard(i)

def run_comprehensive_analysis(sa: StockAnalysis, ticker_index: int) -> None:
    """Run comprehensive analysis for a single ticker."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("="*60 + "\n")
    
    sa.generate_comprehensive_report(ticker_index)

def run_technical_analysis(sa: StockAnalysis, ticker_index: int) -> None:
    """Run technical analysis with charts."""
    print(f"\nGenerating technical analysis charts...")
    sa.plot_technical_analysis(ticker_index)

def display_menu() -> int:
    """Display main menu and get user selection."""
    print("\n" + "="*60)
    print("BE 2100 STOCK ANALYSIS TOOL")
    print("="*60)
    print("\n1. Basic Dashboard (All Stocks)")
    print("2. Comprehensive Report (Single Stock)")
    print("3. Technical Analysis Charts (Single Stock)")
    print("4. Statistical Analysis (Single Stock)")
    print("5. ML Prediction Analysis (Single Stock)")
    print("6. Run All Analyses")
    print("0. Exit")
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = int(input("Enter your choice (0-6): "))
            if 0 <= choice <= 6:
                return choice
            print("Invalid choice. Please enter a number between 0 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_ticker(tickers: List[str]) -> int:
    """Allow user to select a ticker from the list."""
    print("\nAvailable tickers:")
    for i, ticker in enumerate(tickers):
        print(f"{i+1}. {ticker}")
    
    while True:
        try:
            choice = int(input(f"\nSelect ticker (1-{len(tickers)}): ")) - 1
            if 0 <= choice < len(tickers):
                return choice
            print(f"Invalid choice. Please enter a number between 1 and {len(tickers)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """Main application entry point."""
    # Configuration
    tickers = ["NVDA", "AMD", "MSFT", "AAPL"]
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    interval = "1d"
    csv_path = "stock_data.csv"
    
    # Initialize StockAnalysis
    print("\nInitializing Stock Analysis Tool...")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    
    sa = StockAnalysis(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        path=csv_path
    )
    
    # Fetch data
    print("\nFetching stock data...")
    try:
        sa.fetch_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Interactive menu
    while True:
        choice = display_menu()
        
        if choice == 0:
            print("\nThank you for using BE 2100 Stock Analysis Tool!")
            print("Exiting...")
            break
        
        elif choice == 1:
            # Basic Dashboard for all stocks
            run_basic_analysis(sa, tickers)
        
        elif choice == 2:
            # Comprehensive Report
            ticker_idx = select_ticker(tickers)
            run_comprehensive_analysis(sa, ticker_idx)
        
        elif choice == 3:
            # Technical Analysis
            ticker_idx = select_ticker(tickers)
            run_technical_analysis(sa, ticker_idx)
        
        elif choice == 4:
            # Statistical Analysis
            ticker_idx = select_ticker(tickers)
            print(f"\nRunning statistical analysis for {tickers[ticker_idx]}...")
            sa.perform_statistical_analysis(ticker_idx)
        
        elif choice == 5:
            # ML Prediction
            ticker_idx = select_ticker(tickers)
            sa.perform_ml_analysis(ticker_idx)
            sa.predict_future_price(ticker_idx)
        
        elif choice == 6:
            # Run all analyses
            ticker_idx = select_ticker(tickers)
            print(f"\nRunning all analyses for {tickers[ticker_idx]}...")
            
            # Comprehensive report
            run_comprehensive_analysis(sa, ticker_idx)
            
            # Technical charts
            run_technical_analysis(sa, ticker_idx)
            
            # ML analysis
            sa.perform_ml_analysis(ticker_idx)
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()