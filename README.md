# BE 2100 Project: Advanced Stock Market Analysis Tool

## 📊 Project Overview

This project fulfills the requirements for the **BE 2100 Final Project (Option 1: Engineering Problem)** with significant enhancements and additional features.

The objective is to apply comprehensive engineering statistical analysis, machine learning techniques, and technical analysis to historical financial data. This tool analyzes daily price movements of major tech stocks (NVDA, AMD, MSFT, AAPL), investigating whether intra-day metrics and technical indicators are statistically significant predictors of closing prices and future price movements.

## ✨ Features

### Core Statistical Analysis
1. **Data Collection:** Automatically fetches historical data using the Yahoo Finance API (`yfinance`)
2. **Descriptive Statistics:** Generates frequency distributions (Histograms) and outlier detection (Box Plots)
3. **Statistical Intervals:** Calculates 95% Confidence Intervals for population means
4. **Hypothesis Testing:** Performs One-Sample T-tests to determine if daily returns are statistically significant
5. **Machine Learning:** Implements Linear Regression models to predict closing prices with R² accuracy and error metrics

### Enhanced Features (NEW!)
6. **Technical Indicators:**
   - Simple Moving Averages (SMA 20, 50, 200)
   - Exponential Moving Averages (EMA 12, 26)
   - Relative Strength Index (RSI)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands

7. **Risk & Performance Metrics:**
   - Annualized Volatility
   - Sharpe Ratio (risk-adjusted returns)
   - Maximum Drawdown
   - Value at Risk (VaR 95%)
   - Conditional Value at Risk (CVaR)

8. **Advanced Visualizations:**
   - Interactive dashboards
   - Multi-panel technical analysis charts
   - Comprehensive visual reports

9. **Future Price Prediction:**
   - Next-day price predictions
   - Trend analysis
   - Model confidence metrics

10. **Interactive Menu System:**
    - User-friendly command-line interface
    - Multiple analysis modes
    - Flexible ticker selection

## 🔧 Prerequisites

- **Python 3.8+** installed
- Required libraries (see Installation section)

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/Kevin4663/BE2100-Project-1-Group-8.git
cd BE2100-Project-1-Group-8
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Web Application (Recommended) 🌐

Run the professional web interface with Streamlit:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run the web app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

**Features:**
- 📊 Interactive dashboards with real-time filtering
- 📈 Beautiful charts with zoom, pan, and download capabilities
- 🔄 Multi-stock comparison
- 📥 Downloadable reports
- 🎨 Professional, clean UI

### Command-Line Interactive Mode

Run the terminal-based application with an interactive menu:

```bash
python main.py
```

This will present you with a menu:
```
BE 2100 STOCK ANALYSIS TOOL
============================================================

1. Basic Dashboard (All Stocks)
2. Comprehensive Report (Single Stock)
3. Technical Analysis Charts (Single Stock)
4. Statistical Analysis (Single Stock)
5. ML Prediction Analysis (Single Stock)
6. Run All Analyses
0. Exit
```

### Programmatic Usage

You can also use the `StockAnalysis` class directly in your own scripts:

```python
from StockAnalysis import StockAnalysis
from datetime import datetime

# Initialize
tickers = ["AAPL", "MSFT"]
sa = StockAnalysis(
    tickers=tickers,
    start_date="2023-01-01",
    end_date=datetime.today().strftime('%Y-%m-%d'),
    interval="1d",
    path="stock_data.csv"
)

# Fetch data
sa.fetch_data()

# Run various analyses
sa.perform_statistical_analysis(0)  # Statistical analysis for first ticker
sa.plot_technical_analysis(0)       # Technical charts
sa.generate_comprehensive_report(0) # Full report
```

## 📊 Available Analysis Methods

### Statistical Analysis
- `perform_statistical_analysis(ticker_index)`: Generates statistical metrics and hypothesis tests
- `calculate_risk_metrics(ticker_index)`: Computes volatility, Sharpe ratio, VaR, etc.

### Technical Analysis
- `calculate_technical_indicators(ticker_index)`: Computes SMA, EMA, RSI, MACD, Bollinger Bands
- `plot_technical_analysis(ticker_index)`: Visualizes technical indicators

### Machine Learning
- `calculate_ml_model(ticker_index)`: Trains and evaluates linear regression model
- `perform_ml_analysis(ticker_index)`: Displays ML results with visualization
- `predict_future_price(ticker_index)`: Predicts next trading day's closing price

### Visualization
- `plot_histogram(ticker_index, column)`: Frequency distribution plots
- `plot_boxplot(ticker_index, column)`: Outlier detection plots
- `plot_dashboard(ticker_index)`: Comprehensive 4-panel dashboard

### Comprehensive Reporting
- `generate_comprehensive_report(ticker_index)`: Full analysis report with all metrics

## 📈 Example Output

### Statistical Analysis Output
```
STATISTICAL ANALYSIS
------------------------------------------------------------
Statistical Analysis for AAPL
Mean Close Price: $178.45
95% Confidence Interval: $175.32 to $181.58

Hypothesis: Is the daily return significantly different from 0
Hypothesis Test (T-Test) P-Value: 0.12345
Conclusion: Fail to Reject Null (Price movement is random)
```

### Risk Metrics Output
```
RISK & PERFORMANCE METRICS
------------------------------------------------------------
Annual Volatility: 28.45%
Sharpe Ratio: 0.8234
Maximum Drawdown: -15.67%
Value at Risk (95%): -2.34%
Conditional VaR (95%): -3.12%
Annualized Return: 18.92%
```

## 🏗️ Project Structure

```
BE2100-Project-1-Group-8/
├── app.py                    # 🌐 Web application (Streamlit)
├── StockAnalysis.py          # 📊 Main analysis class with all methods
├── main.py                   # 💻 Command-line interactive application
├── test_improvements.py      # 🧪 Test suite for validation
├── requirements.txt          # 📦 Python dependencies
├── README.md                 # 📖 This file
├── .gitignore               # 🚫 Git ignore rules
├── venv/                     # 🐍 Virtual environment (excluded from git)
└── stock_data.csv           # 💾 Generated data file (after first run)
```

## 🔬 Technical Details

### Machine Learning Model
- **Algorithm:** Linear Regression (scikit-learn)
- **Features:** Open, High, Low, Volume
- **Target:** Close price
- **Train/Test Split:** 80/20
- **Metrics:** R² Score, MSE, MAE

### Technical Indicators
- **Moving Averages:** SMA (20, 50, 200), EMA (12, 26)
- **Momentum:** RSI (14-period)
- **Trend:** MACD (12, 26, 9)
- **Volatility:** Bollinger Bands (20-period, 2 std dev)

### Statistical Methods
- **Confidence Intervals:** t-distribution based (95% confidence)
- **Hypothesis Testing:** One-sample t-test
- **Risk Metrics:** Annualized volatility (√252 scaling)

## 🎯 Code Improvements (Enhanced Version)

This enhanced version includes:

1. **Type Hints:** Full type annotations for better code clarity
2. **Comprehensive Docstrings:** Detailed documentation for all methods
3. **Error Handling:** Try-catch blocks for robust operation
4. **Code Quality:** Fixed typos (plot_botplot → plot_boxplot)
5. **Enhanced Dependencies:** Added missing packages with version specifications
6. **New Features:** Technical indicators, risk metrics, comprehensive reporting
7. **Interactive UI:** Menu-driven interface for ease of use
8. **Better Visualization:** Multi-panel charts with professional styling

## 📝 Dependencies

- `yfinance>=0.2.28` - Yahoo Finance data fetching
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `scipy>=1.10.0` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning
- `ta>=0.11.0` - Technical analysis indicators
- `plotly>=5.14.0` - Interactive visualizations
- `streamlit>=1.28.0` - Web application framework

## 👥 Contributors

- **Original Project:** Group 8, BE 2100
- **Enhanced Version:** Yosef (Branch: Yosef)

## 📄 License

This project is for educational purposes as part of the BE 2100 course.

## 🙏 Acknowledgments

- Yahoo Finance API for providing free stock data
- BE 2100 course instructors and teaching staff
- Open-source Python data science community

## 📧 Contact

For questions or feedback about this enhanced version, please contact through the repository issues page.

---

**Note:** This tool is for educational and analytical purposes only. Not financial advice.
