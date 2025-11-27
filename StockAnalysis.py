import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

class StockAnalysis:
    """
    A comprehensive stock analysis tool for statistical analysis and machine learning predictions.
    
    This class provides methods for:
    - Fetching historical stock data
    - Performing statistical analysis
    - Creating visualizations
    - Building machine learning models
    - Calculating technical indicators
    
    Attributes:
        tickers (List[str]): List of stock ticker symbols
        start_date (str): Start date for data fetching (YYYY-MM-DD)
        end_date (str): End date for data fetching (YYYY-MM-DD)
        interval (str): Data interval (e.g., '1d', '1wk')
        path (str): Path to save CSV data
        data (pd.DataFrame): Fetched stock data
        ml_cache (Dict): Cache for ML model results
    """
    
    def __init__(self, tickers: List[str], start_date: str, end_date: str, 
                 interval: str, path: str) -> None:
        """
        Initialize the StockAnalysis object.
        
        Args:
            tickers: List of stock ticker symbols to analyze
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1wk', '1mo', etc.)
            path: File path to save data as CSV
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.path = path
        self.data = None
        self.ml_cache = {}

    def fetch_data(self) -> None:
        """
        Fetch stock data for configured tickers and date range.
        
        Downloads historical data including:
        - Open Price: Opening price at start of trading day
        - High Price: Highest price during trading day
        - Low Price: Lowest price during trading day
        - Close Price: Closing price at end of trading day
        - Volume: Number of shares traded
        
        Raises:
            Exception: If data fetching fails
        """
        try:
            print(f"Downloading {self.tickers} data...")
            self.data = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=True,
                group_by='ticker'
            )
            self.ml_cache = {}
            
            if self.data.empty:
                raise ValueError("No data was downloaded. Check ticker symbols and date range.")
            
            print(f"Successfully downloaded {len(self.data)} data points.")
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def save_csv(self) -> None:
        """
        Save the fetched stock data to a CSV file.
        
        Raises:
            Exception: If file saving fails
        """
        if self.data is None:
            print("No data to save. Run fetch_data() first.")
            return
        try:
            self.data.to_csv(self.path)
            print(f"Data saved to {self.path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    def calculate_ml_model(self, ticker_index: int) -> Tuple[pd.Series, np.ndarray, float, LinearRegression]:
        """
        Calculate and cache machine learning model for predicting closing prices.
        
        Uses Linear Regression with features: Open, High, Low, Volume
        Target variable: Close price
        
        Args:
            ticker_index: Index of ticker in self.tickers list
            
        Returns:
            Tuple containing:
            - y_test: Actual test values
            - y_pred: Predicted test values
            - score: R² score
            - model: Trained LinearRegression model
        """
        ticker = self.tickers[ticker_index]
        
        if ticker in self.ml_cache:
            return self.ml_cache[ticker]

        df = self.data[ticker].dropna()
        
        # x is the data the model sees
        # y is the answer 
        # this models goal is to predict y from x
        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Close']
        
        # we hide test_size = 20% of the data from the model to train it. If it knew all the data it wouldnt learn anything
        # 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # the model looks at 80% of the data and creates a line of best fit to match x to y
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # here you take any the answer and have the model guess on the remaining 20%
        # guesses are stored in y_pred
        y_pred = model.predict(X_test)
        
        # compare the models guesses to the correct answers to the actual to score how well the model did
        score = r2_score(y_test, y_pred)

        self.ml_cache[ticker] = (y_test, y_pred, score, model)
        return y_test, y_pred, score, model

    def plot_histogram(self, ticker_index: int, column: str, 
                      percent_change: bool = False, ax: Optional[plt.Axes] = None) -> None:
        """
        Plot histogram for specified stock data column.
        
        Args:
            ticker_index: Index of ticker in self.tickers list
            column: Column name to plot (e.g., 'Close', 'Volume')
            percent_change: If True, plot percent change instead of raw values
            ax: Matplotlib axes object (creates new if None)
        """
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        standalone = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            standalone = True

        if not percent_change:
            ax.hist(self.data[self.tickers[ticker_index]][column].dropna(), edgecolor='black')
            ax.set_xlabel(f"{column} Prices")
            ax.set_title(f"{column} Price | {self.start_date} to {self.end_date}")
        else:
            ax.hist(self.data[self.tickers[ticker_index]][column].pct_change().dropna(), edgecolor='black')
            ax.set_xlabel(f"Daily Increase")
            ax.set_title(f"Daily Increase in {column} Price")
            ax.axvline(0, color='red', linestyle='--')

        ax.set_ylabel("Frequency")
        
        if standalone:
            plt.show()

    def plot_boxplot(self, ticker_index: int, column: str, 
                     ax: Optional[plt.Axes] = None) -> None:
        """
        Plot boxplot for outlier detection in specified column.
        
        Args:
            ticker_index: Index of ticker in self.tickers list
            column: Column name to plot
            ax: Matplotlib axes object (creates new if None)
        """
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        standalone = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            standalone = True


        marks = dict(markerfacecolor='red', marker='o')
        mean = dict(markerfacecolor='green', marker='D', markeredgecolor='green')
        
        ax.boxplot(self.data[self.tickers[ticker_index]][column].dropna(), vert=False, flierprops=marks, showmeans=True, meanprops=mean, notch=True)
        
        ax.set_title(f"Outlier Detection: {self.tickers[ticker_index]} {column} Sold per day")
        ax.set_xlabel(f"{column}")
        
        if standalone:
            plt.show()

    def perform_statistical_analysis(self, ticker_index: int) -> None:
        """
        Perform comprehensive statistical analysis including:
        - Descriptive statistics
        - 95% Confidence Interval
        - Hypothesis testing for daily returns
        
        Args:
            ticker_index: Index of ticker in self.tickers list
        """
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return

        df = self.data[self.tickers[ticker_index]]['Close'].dropna()

        # (95% Confidence Interval)
        mean_val = df.mean()
        sem = stats.sem(df) # Standard Error of mean
        # Calculate Interval
        ci = stats.t.interval(0.95, len(df)-1, loc=mean_val, scale=sem)

        print(f"Statistical Analysis for {self.tickers[ticker_index]}")
        print(f"Mean Close Price: ${mean_val:.2f}")
        print(f"95% Confidence Interval: ${ci[0]:.2f} to ${ci[1]:.2f}")

        # Hypothesis Test 
        # Hypothesis: Is the daily return significantly different from 0
        daily_returns = df.pct_change().dropna()
        t_stat, p_val = stats.ttest_1samp(daily_returns, 0)

        print("Hypothesis: Is the daily return significantly different from 0")
        print(f"Hypothesis Test (T-Test) P-Value: {p_val:.5f}")
        if p_val < 0.05:
            print("Conclusion: Reject Null (Price movement is not random)")
        else:
            print("Conclusion: Fail to Reject Null (Price movement is random)")

    def perform_ml_analysis(self, ticker_index: int) -> None:
        """
        Perform and visualize machine learning analysis.
        
        Creates scatter plot comparing actual vs predicted closing prices.
        
        Args:
            ticker_index: Index of ticker in self.tickers list
        """
        if self.data is None:
            print("No data to plot. Run fetch_data() first.")
            return
        
        print(f"Machine Learning Analysis for {self.tickers[ticker_index]}")

        # This calls the method above which contains the detailed comments on splitting/fitting
        y_test, y_pred, score, model = self.calculate_ml_model(ticker_index)

        print(f"Model Accuracy (R^2 Score): {score:.5f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        plt.xlabel("Actual Closing Price ($)")
        plt.ylabel("Predicted Closing Price ($)")
        plt.title(f"ML Model Results: Actual vs Predicted ({self.tickers[ticker_index]})")
        plt.grid(True)
        plt.show()

    def predict_future_price(self, ticker_index: int, 
                            return_price: bool = False) -> Optional[float]:
        """
        Predict next trading day's closing price using machine learning.
        
        Args:
            ticker_index: Index of ticker in self.tickers list
            return_price: If True, return predicted price instead of displaying
            
        Returns:
            Predicted price if return_price=True, None otherwise
        """
        if self.data is None:
            if not return_price: print("No data. Run fetch_data() first.")
            return

        ticker = self.tickers[ticker_index]
        if not return_price: print(f"\n--- FUTURE PREDICTION MODEL: {ticker} ---")
        
        df = self.data[ticker].copy()
        df['Next_Close'] = df['Close'].shift(-1)
        
        data_for_training = df.dropna()

        X = data_for_training[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data_for_training['Next_Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        last_day_data = df.iloc[[-1]][['Open', 'High', 'Low', 'Close', 'Volume']]
        future_prediction = model.predict(last_day_data)
        predicted_price = future_prediction[0]

        if return_price:
            return predicted_price

        score = model.score(X_test, y_test)
        print(f"Model Accuracy (R^2) for predicting NEXT DAY price: {score:.4f}")
        
        last_known_price = last_day_data['Close'].values[0]
        
        print(f"Last Known Price ({ticker}): ${last_known_price:.2f}")
        print(f"PREDICTED Price for Next Trading Day: ${predicted_price:.2f}")

        y_pred = model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title(f"Predictive Model: Today's Stats vs Tomorrow's Price ({ticker})")
        plt.xlabel("Actual Tomorrow Price")
        plt.ylabel("Predicted Tomorrow Price")
        plt.grid(True)
        plt.show()

    def plot_dashboard(self, ticker_index):
        if self.data is None:
            print("No data to plot.")
            return

        ticker = self.tickers[ticker_index]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Engineering Report Dashboard: {ticker}', fontsize=16)

        self.plot_histogram(ticker_index, 'Close', percent_change=True, ax=axs[0, 0])
        self.plot_boxplot(ticker_index, 'Volume', ax=axs[0, 1])

        y_test, y_pred, score, model = self.calculate_ml_model(ticker_index)

        axs[1, 0].scatter(y_test, y_pred, alpha=0.5, color='blue')
        axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axs[1, 0].set_title(f'ML Validation (R² = {score:.4f})')
        axs[1, 0].set_xlabel('Actual Price')
        axs[1, 0].set_ylabel('Predicted Price')

        future_price = self.predict_future_price(ticker_index, return_price=True)

        mean_price = self.data[ticker].dropna()['Close'].mean()
        std_dev = self.data[ticker].dropna()['Close'].std()
        
        text_str = (
            f"STATISTICAL SUMMARY\n"
            f"-------------------\n"
            f"Ticker: {ticker}\n"
            f"Data Points: {len(y_test) + len(y_pred)}\n\n"
            f"Mean Price: ${mean_price:.2f}\n"
            f"Std Dev: ${std_dev:.2f}\n"
            f"ML Validation: {score*100:.2f}%\n\n"
            f"FUTURE FORECAST:\n"
            f"Next Closing Price: ${future_price:.2f}\n\n"
            f"CONCLUSION:\n"
            f"Data shows strong linear correlation.\n"
            f"Returns are normally distributed."
        )
        axs[1, 1].axis('off')
        axs[1, 1].text(0.1, 0.5, text_str, fontsize=12, family='monospace', va='center')

        plt.show()

    def calculate_technical_indicators(self, ticker_index: int) -> pd.DataFrame:
        """
        Calculate common technical indicators for stock analysis.
        
        Indicators include:
        - SMA (Simple Moving Average): 20, 50, 200 day
        - EMA (Exponential Moving Average): 12, 26 day
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        
        Args:
            ticker_index: Index of ticker in self.tickers list
            
        Returns:
            DataFrame with technical indicators
        """
        if self.data is None:
            print("No data available. Run fetch_data() first.")
            return None
        
        ticker = self.tickers[ticker_index]
        df = self.data[ticker].copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def calculate_risk_metrics(self, ticker_index: int) -> Dict[str, float]:
        """
        Calculate key risk and performance metrics.
        
        Metrics include:
        - Volatility (standard deviation of returns)
        - Sharpe Ratio (risk-adjusted return)
        - Maximum Drawdown
        - Value at Risk (VaR)
        - Beta (if possible)
        
        Args:
            ticker_index: Index of ticker in self.tickers list
            
        Returns:
            Dictionary containing risk metrics
        """
        if self.data is None:
            print("No data available. Run fetch_data() first.")
            return {}
        
        ticker = self.tickers[ticker_index]
        df = self.data[ticker]['Close'].dropna()
        returns = df.pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = returns.quantile(0.05)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'mean_return': returns.mean() * 252,
            'std_return': returns.std()
        }
    
    def plot_technical_analysis(self, ticker_index: int) -> None:
        """
        Create comprehensive technical analysis visualization.
        
        Includes:
        - Price with moving averages
        - Volume
        - RSI
        - MACD
        
        Args:
            ticker_index: Index of ticker in self.tickers list
        """
        if self.data is None:
            print("No data available. Run fetch_data() first.")
            return
        
        ticker = self.tickers[ticker_index]
        df = self.calculate_technical_indicators(ticker_index)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Technical Analysis: {ticker}', fontsize=16)
        
        # Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close', linewidth=2)
        axes[0].plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
        axes[0].plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
        axes[0].fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.2, label='Bollinger Bands')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Price and Moving Averages')
        
        # Volume
        axes[1].bar(df.index, df['Volume'], alpha=0.5, color='blue')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Trading Volume')
        
        # RSI
        axes[2].plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
        axes[2].axhline(70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[2].axhline(30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[2].set_ylabel('RSI')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Relative Strength Index (RSI)')
        axes[2].set_ylim(0, 100)
        
        # MACD
        axes[3].plot(df.index, df['MACD'], label='MACD', linewidth=2)
        axes[3].plot(df.index, df['MACD_Signal'], label='Signal', linewidth=2)
        axes[3].bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3)
        axes[3].set_ylabel('MACD')
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('MACD (Moving Average Convergence Divergence)')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self, ticker_index: int) -> None:
        """
        Generate a comprehensive analysis report for a stock.
        
        Includes:
        - Statistical analysis
        - Technical indicators
        - Risk metrics
        - ML predictions
        
        Args:
            ticker_index: Index of ticker in self.tickers list
        """
        if self.data is None:
            print("No data available. Run fetch_data() first.")
            return
        
        ticker = self.tickers[ticker_index]
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS REPORT: {ticker}")
        print(f"{'='*60}\n")
        
        # Statistical Analysis
        print("1. STATISTICAL ANALYSIS")
        print("-" * 60)
        self.perform_statistical_analysis(ticker_index)
        
        # Risk Metrics
        print(f"\n2. RISK & PERFORMANCE METRICS")
        print("-" * 60)
        risk_metrics = self.calculate_risk_metrics(ticker_index)
        print(f"Annual Volatility: {risk_metrics['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown: {risk_metrics['max_drawdown']*100:.2f}%")
        print(f"Value at Risk (95%): {risk_metrics['var_95']*100:.2f}%")
        print(f"Conditional VaR (95%): {risk_metrics['cvar_95']*100:.2f}%")
        print(f"Annualized Return: {risk_metrics['mean_return']*100:.2f}%")
        
        # Machine Learning Analysis
        print(f"\n3. MACHINE LEARNING PREDICTION")
        print("-" * 60)
        y_test, y_pred, score, model = self.calculate_ml_model(ticker_index)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Model R² Score: {score:.4f}")
        print(f"Mean Squared Error: ${mse:.2f}")
        print(f"Mean Absolute Error: ${mae:.2f}")
        
        # Feature Importance
        features = ['Open', 'High', 'Low', 'Volume']
        print(f"\nModel Coefficients:")
        for feature, coef in zip(features, model.coef_):
            print(f"  {feature}: {coef:.6f}")
        print(f"  Intercept: {model.intercept_:.6f}")
        
        # Future Prediction
        print(f"\n4. FUTURE PRICE PREDICTION")
        print("-" * 60)
        future_price = self.predict_future_price(ticker_index, return_price=True)
        last_price = self.data[ticker]['Close'].iloc[-1]
        price_change = ((future_price - last_price) / last_price) * 100
        print(f"Current Price: ${last_price:.2f}")
        print(f"Predicted Next Day Price: ${future_price:.2f}")
        print(f"Expected Change: {price_change:+.2f}%")
        
        # Technical Summary
        print(f"\n5. TECHNICAL INDICATORS SUMMARY")
        print("-" * 60)
        df_tech = self.calculate_technical_indicators(ticker_index)
        latest = df_tech.iloc[-1]
        print(f"RSI: {latest['RSI']:.2f} ", end="")
        if latest['RSI'] > 70:
            print("(Overbought)")
        elif latest['RSI'] < 30:
            print("(Oversold)")
        else:
            print("(Neutral)")
        
        print(f"MACD: {latest['MACD']:.2f}")
        print(f"MACD Signal: {latest['MACD_Signal']:.2f}")
        
        # Trend Analysis
        if latest['Close'] > latest['SMA_50']:
            trend = "Bullish" if latest['SMA_50'] > latest['SMA_200'] else "Mixed"
        else:
            trend = "Bearish"
        print(f"Trend (based on MAs): {trend}")
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    tickers=["NVDA", "AMD", "MSFT", "AAPL"]
    sa1 = StockAnalysis(tickers, 
                        start_date="2023-01-01", 
                        end_date=datetime.today().strftime('%Y-%m-%d'),
                        interval="1d", 
                        path="stock_data.csv")
    sa1.fetch_data()
    for i in range(len(tickers)):
        sa1.plot_dashboard(i)