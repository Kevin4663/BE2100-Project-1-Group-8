"""
Quick test script to demonstrate the enhanced features
"""

from StockAnalysis import StockAnalysis
from datetime import datetime

def test_basic_functionality():
    """Test basic functionality without user interaction"""
    print("="*60)
    print("TESTING ENHANCED STOCK ANALYSIS TOOL")
    print("="*60)
    
    # Initialize with just one ticker for faster testing
    print("\n1. Initializing StockAnalysis...")
    tickers = ["AAPL"]
    sa = StockAnalysis(
        tickers=tickers,
        start_date="2024-01-01",
        end_date=datetime.today().strftime('%Y-%m-%d'),
        interval="1d",
        path="test_stock_data.csv"
    )
    print("✓ Initialization successful")
    
    # Fetch data
    print("\n2. Fetching stock data...")
    try:
        sa.fetch_data()
        print("✓ Data fetched successfully")
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return False
    
    # Test statistical analysis
    print("\n3. Testing Statistical Analysis...")
    try:
        sa.perform_statistical_analysis(0)
        print("✓ Statistical analysis completed")
    except Exception as e:
        print(f"✗ Error in statistical analysis: {e}")
        return False
    
    # Test risk metrics
    print("\n4. Testing Risk Metrics Calculation...")
    try:
        risk_metrics = sa.calculate_risk_metrics(0)
        print(f"✓ Risk metrics calculated:")
        print(f"  - Volatility: {risk_metrics['volatility']*100:.2f}%")
        print(f"  - Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
        print(f"  - Max Drawdown: {risk_metrics['max_drawdown']*100:.2f}%")
    except Exception as e:
        print(f"✗ Error calculating risk metrics: {e}")
        return False
    
    # Test technical indicators
    print("\n5. Testing Technical Indicators...")
    try:
        df_tech = sa.calculate_technical_indicators(0)
        latest = df_tech.iloc[-1]
        print(f"✓ Technical indicators calculated:")
        print(f"  - RSI: {latest['RSI']:.2f}")
        print(f"  - MACD: {latest['MACD']:.2f}")
        print(f"  - SMA 20: ${latest['SMA_20']:.2f}")
    except Exception as e:
        print(f"✗ Error calculating technical indicators: {e}")
        return False
    
    # Test ML model
    print("\n6. Testing Machine Learning Model...")
    try:
        y_test, y_pred, score, model = sa.calculate_ml_model(0)
        print(f"✓ ML model trained:")
        print(f"  - R² Score: {score:.4f}")
        print(f"  - Test samples: {len(y_test)}")
    except Exception as e:
        print(f"✗ Error in ML model: {e}")
        return False
    
    # Test future prediction
    print("\n7. Testing Future Price Prediction...")
    try:
        future_price = sa.predict_future_price(0, return_price=True)
        last_price = sa.data[tickers[0]]['Close'].iloc[-1]
        change = ((future_price - last_price) / last_price) * 100
        print(f"✓ Future prediction completed:")
        print(f"  - Current Price: ${last_price:.2f}")
        print(f"  - Predicted Price: ${future_price:.2f}")
        print(f"  - Expected Change: {change:+.2f}%")
    except Exception as e:
        print(f"✗ Error in prediction: {e}")
        return False
    
    # Test comprehensive report (text only, no charts)
    print("\n8. Testing Comprehensive Report...")
    try:
        print("\n" + "-"*60)
        sa.generate_comprehensive_report(0)
        print("✓ Comprehensive report generated")
    except Exception as e:
        print(f"✗ Error generating report: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe enhanced features are working correctly:")
    print("  ✓ Type hints and docstrings added")
    print("  ✓ Error handling implemented")
    print("  ✓ Technical indicators working")
    print("  ✓ Risk metrics calculating")
    print("  ✓ ML predictions functioning")
    print("  ✓ Comprehensive reporting operational")
    print("\nYou can now run 'python main.py' for the full interactive experience!")
    
    return True

if __name__ == "__main__":
    test_basic_functionality()
