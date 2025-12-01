"""
BE 2100 Stock Analysis - Web Application
Professional frontend using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from StockAnalysis import StockAnalysis

# Page configuration
st.set_page_config(
    page_title="BE 2100 Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data(tickers, start_date, end_date, interval):
    """Load and cache stock data"""
    sa = StockAnalysis(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        path="stock_data.csv"
    )
    sa.fetch_data()
    return sa

def create_price_chart(sa, ticker_index):
    """Create interactive price chart with volume"""
    ticker = sa.tickers[ticker_index]
    df = sa.data[ticker].copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_technical_chart(sa, ticker_index):
    """Create technical indicators chart"""
    ticker = sa.tickers[ticker_index]
    df = sa.calculate_technical_indicators(ticker_index)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} with Moving Averages', 'RSI', 'MACD'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and MAs
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                            line=dict(color='gray', dash='dash'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                            line=dict(color='gray', dash='dash'), fill='tonexty', opacity=0.2), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def create_ml_scatter(sa, ticker_index):
    """Create ML prediction scatter plot"""
    y_test, y_pred, score, model = sa.calculate_ml_model(ticker_index)
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'ML Model: Actual vs Predicted (R² = {score:.4f})',
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        height=500
    )
    
    return fig

def main():
    # Header
    st.title("BE 2100 Stock Market Analysis Tool")
    st.markdown("**Professional Stock Analysis with Statistical Methods, Machine Learning & Technical Indicators**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Ticker selection
        default_tickers = ["NVDA", "AMD", "MSFT", "AAPL"]
        tickers_input = st.text_input(
            "Stock Tickers (comma-separated)",
            value=", ".join(default_tickers)
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2024-01-01")
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime(datetime.today())
            )
        
        # Interval
        interval = st.selectbox(
            "Interval",
            options=["1d", "1wk", "1mo"],
            index=0
        )
        
        # Load data button
        if st.button("Load Data", type="primary", use_container_width=True):
            st.session_state.data_loaded = False

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool provides:
        - Statistical Analysis
        - Risk Metrics
        - Technical Indicators
        - ML Predictions
        - Future Price Forecasts
        """)
    
    # Load data
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    try:
        with st.spinner("Loading stock data..."):
            sa = load_stock_data(
                tickers,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                interval
            )
        st.session_state.sa = sa
        st.session_state.data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    if not st.session_state.data_loaded:
        st.stop()
    
    sa = st.session_state.sa
    
    # Ticker selector
    st.sidebar.markdown("---")
    selected_ticker = st.sidebar.selectbox(
        "Select Stock to Analyze",
        options=tickers,
        index=0
    )
    ticker_index = tickers.index(selected_ticker)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Technical Analysis",
        "ML Predictions",
        "Risk Metrics",
        "Full Report"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header(f"{selected_ticker} Overview")
        
        # Key metrics
        ticker_data = sa.data[selected_ticker]['Close'].dropna()
        current_price = ticker_data.iloc[-1]
        price_change = ticker_data.iloc[-1] - ticker_data.iloc[-2]
        price_change_pct = (price_change / ticker_data.iloc[-2]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
        with col2:
            st.metric("Mean Price", f"${ticker_data.mean():.2f}")
        with col3:
            st.metric("High (Period)", f"${ticker_data.max():.2f}")
        with col4:
            st.metric("Low (Period)", f"${ticker_data.min():.2f}")
        
        # Price chart
        st.subheader("Price & Volume Chart")
        fig = create_price_chart(sa, ticker_index)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            df = sa.data[selected_ticker]['Close'].dropna()
            mean_val = df.mean()
            from scipy import stats
            sem = stats.sem(df)
            ci = stats.t.interval(0.95, len(df)-1, loc=mean_val, scale=sem)
            
            st.markdown(f"""
            **Confidence Interval (95%)**
            - Lower Bound: ${ci[0]:.2f}
            - Mean: ${mean_val:.2f}
            - Upper Bound: ${ci[1]:.2f}
            """)
        
        with col2:
            daily_returns = df.pct_change().dropna()
            t_stat, p_val = stats.ttest_1samp(daily_returns, 0)
            
            st.markdown(f"""
            **Hypothesis Test (Daily Returns ≠ 0)**
            - T-Statistic: {t_stat:.4f}
            - P-Value: {p_val:.5f}
            - Result: {'Reject Null' if p_val < 0.05 else 'Fail to Reject Null'}
            """)
    
    # Tab 2: Technical Analysis
    with tab2:
        st.header(f"{selected_ticker} Technical Analysis")
        
        df_tech = sa.calculate_technical_indicators(ticker_index)
        latest = df_tech.iloc[-1]
        
        # Technical indicators metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi_status = "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral"
            st.metric("RSI", f"{latest['RSI']:.2f}", rsi_status)
        with col2:
            st.metric("MACD", f"{latest['MACD']:.2f}")
        with col3:
            st.metric("SMA 50", f"${latest['SMA_50']:.2f}")
        with col4:
            trend = "Bullish" if latest['Close'] > latest['SMA_50'] else "Bearish"
            st.metric("Trend", trend)
        
        # Technical chart
        fig = create_technical_chart(sa, ticker_index)
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicator explanations
        with st.expander("Technical Indicators Explained"):
            st.markdown("""
            **RSI (Relative Strength Index)**
            - Above 70: Overbought (potential sell signal)
            - Below 30: Oversold (potential buy signal)
            
            **MACD (Moving Average Convergence Divergence)**
            - MACD above Signal: Bullish momentum
            - MACD below Signal: Bearish momentum
            
            **Bollinger Bands**
            - Price at upper band: Potentially overbought
            - Price at lower band: Potentially oversold
            """)
    
    # Tab 3: ML Predictions
    with tab3:
        st.header(f"{selected_ticker} Machine Learning Predictions")
        
        y_test, y_pred, score, model = sa.calculate_ml_model(ticker_index)
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # ML metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{score:.4f}", "Higher is better")
        with col2:
            st.metric("MSE", f"${mse:.2f}")
        with col3:
            st.metric("MAE", f"${mae:.2f}")
        
        # ML scatter plot
        fig = create_ml_scatter(sa, ticker_index)
        st.plotly_chart(fig, use_container_width=True)
        
        # Future prediction
        st.subheader("Next Day Price Prediction")
        future_price = sa.predict_future_price(ticker_index, return_price=True)
        last_price = sa.data[selected_ticker]['Close'].iloc[-1]
        price_change = ((future_price - last_price) / last_price) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${last_price:.2f}")
        with col2:
            st.metric("Predicted Price", f"${future_price:.2f}")
        with col3:
            delta_color = "normal" if abs(price_change) < 2 else "inverse"
            st.metric("Expected Change", f"{price_change:+.2f}%", delta_color=delta_color)
        
        # Model coefficients
        with st.expander("Model Details"):
            features = ['Open', 'High', 'Low', 'Volume']
            coef_df = pd.DataFrame({
                'Feature': features + ['Intercept'],
                'Coefficient': list(model.coef_) + [model.intercept_]
            })
            st.dataframe(coef_df, use_container_width=True)
    
    # Tab 4: Risk Metrics
    with tab4:
        st.header(f"{selected_ticker} Risk & Performance Metrics")
        
        risk_metrics = sa.calculate_risk_metrics(ticker_index)
        
        # Risk metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            st.metric("Annual Return", f"{risk_metrics['mean_return']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.4f}")
            st.markdown("""
            *Sharpe Ratio measures risk-adjusted return. Higher is better.*
            - > 1.0: Good
            - > 2.0: Very Good
            - > 3.0: Excellent
            """)
        
        with col2:
            st.subheader("Risk Metrics")
            st.metric("Annual Volatility", f"{risk_metrics['volatility']*100:.2f}%")
            st.metric("Maximum Drawdown", f"{risk_metrics['max_drawdown']*100:.2f}%")
            st.metric("VaR (95%)", f"{risk_metrics['var_95']*100:.2f}%")
            st.metric("CVaR (95%)", f"{risk_metrics['cvar_95']*100:.2f}%")
        
        # Risk visualization
        st.subheader("Returns Distribution")
        ticker_data = sa.data[selected_ticker]['Close'].dropna()
        returns = ticker_data.pct_change().dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Daily Returns'))
        fig.add_vline(x=returns.mean(), line_dash="dash", line_color="red", 
                     annotation_text="Mean")
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Full Report
    with tab5:
        st.header(f"{selected_ticker} Comprehensive Report")
        
        # Generate downloadable report
        report_text = f"""
BE 2100 STOCK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {selected_ticker}
{'='*60}

1. PRICE SUMMARY
Current Price: ${current_price:.2f}
Period High: ${ticker_data.max():.2f}
Period Low: ${ticker_data.min():.2f}
Mean Price: ${ticker_data.mean():.2f}

2. STATISTICAL ANALYSIS
95% Confidence Interval: ${ci[0]:.2f} to ${ci[1]:.2f}
Hypothesis Test P-Value: {p_val:.5f}

3. RISK & PERFORMANCE
Annual Return: {risk_metrics['mean_return']*100:.2f}%
Volatility: {risk_metrics['volatility']*100:.2f}%
Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}
Maximum Drawdown: {risk_metrics['max_drawdown']*100:.2f}%
VaR (95%): {risk_metrics['var_95']*100:.2f}%

4. TECHNICAL INDICATORS
RSI: {latest['RSI']:.2f}
MACD: {latest['MACD']:.2f}
SMA 50: ${latest['SMA_50']:.2f}

5. MACHINE LEARNING
Model R² Score: {score:.4f}
Mean Absolute Error: ${mae:.2f}
Predicted Next Price: ${future_price:.2f}
Expected Change: {price_change:+.2f}%
        """
        
        st.text_area("Full Report", report_text, height=400)
        
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name=f"{selected_ticker}_analysis_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
