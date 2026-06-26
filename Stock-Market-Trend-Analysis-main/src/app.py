import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Trend Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Dark Mode Custom Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Main body settings */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Metric Card styling */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #3b82f6;
        }
        
        /* Customize standard streamlit divs */
        .stButton>button {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.00001)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(show_spinner=False)
def load_default_data():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "netflix_stock_5y.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df, "Netflix (NFLX) - Preloaded 5-Year Data"
    return None, None

def fetch_live_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        if df.empty:
            return None
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return None

# --- Header Section ---
st.title("📈 Stock Market Trend Analysis & Prediction")
st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("Settings & Ticker Selection")

data_source = st.sidebar.radio(
    "Data Source",
    ["Use Preloaded Netflix CSV Data", "Fetch Live Ticker from yfinance"]
)

ticker_symbol = "NFLX"
df = None
status_msg = ""

if data_source == "Use Preloaded Netflix CSV Data":
    df, status_msg = load_default_data()
    if df is None:
        st.sidebar.warning("Default Netflix CSV not found. Please select Live Ticker option.")
else:
    ticker_input = st.sidebar.text_input("Enter Ticker Symbol (e.g. AAPL, TSLA, MSFT, NFLX)", value="NFLX")
    if ticker_input:
        with st.spinner(f"Fetching data for {ticker_input}..."):
            df = fetch_live_data(ticker_input.upper().strip())
            if df is not None:
                ticker_symbol = ticker_input.upper().strip()
                status_msg = f"{ticker_symbol} - Real-time Live 5-Year Data"
            else:
                st.sidebar.error(f"Failed to fetch data for '{ticker_input}'. Please check the symbol.")

if df is not None:
    # --- Feature Engineering ---
    # Create copy to prevent modification issues
    df = df.copy()
    
    # Calculate indicators
    df['50_day_MA'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df)
    
    # Calculate Lag Features
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_5'] = df['Close'].shift(5)
    
    # Clean data (drop NaNs from MAs and Lags)
    clean_df = df.dropna().reset_index(drop=True)
    
    st.subheader(f"Dashboard: {status_msg}")
    
    # --- Main Metrics Cards ---
    col1, col2, col3, col4 = st.columns(4)
    latest_row = clean_df.iloc[-1]
    prev_row = clean_df.iloc[-2]
    
    price_change = latest_row['Close'] - prev_row['Close']
    pct_change = (price_change / prev_row['Close']) * 100
    
    col1.metric("Latest Close Price", f"${latest_row['Close']:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("50-Day Moving Average", f"${latest_row['50_day_MA']:.2f}")
    col3.metric("RSI (14-Day)", f"{latest_row['RSI']:.1f}")
    col4.metric("Trading Volume", f"{latest_row['Volume']:,}")
    
    # --- Chronological Train-Test Split ---
    train_data, test_data = train_test_split(clean_df, test_size=0.2, shuffle=False)
    
    features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', '50_day_MA', 'RSI', 'Close_Lag_1', 'Close_Lag_5']
    target = 'Close'
    
    # --- Model Training Section ---
    with st.spinner("Training Models..."):
        # 1. ARIMA Model
        # fit on training target
        try:
            arima_model = ARIMA(train_data[target], order=(5, 1, 1))
            arima_fit = arima_model.fit()
            arima_preds = arima_fit.forecast(steps=len(test_data)).values
        except Exception as e:
            st.error(f"ARIMA training failed: {e}")
            arima_preds = np.zeros(len(test_data))

        # 2. XGBoost Model
        try:
            train_dmatrix = xgb.DMatrix(data=train_data[features], label=train_data[target])
            test_dmatrix = xgb.DMatrix(data=test_data[features], label=test_data[target])
            
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,
                'learning_rate': 0.1
            }
            xgb_model = xgb.train(params=params, dtrain=train_dmatrix)
            xgb_preds = xgb_model.predict(test_dmatrix)
        except Exception as e:
            st.error(f"XGBoost training failed: {e}")
            xgb_preds = np.zeros(len(test_data))

    # --- Calculations of Metrics ---
    # ARIMA Metrics
    mse_arima = mean_squared_error(test_data[target], arima_preds)
    mae_arima = mean_absolute_error(test_data[target], arima_preds)
    r2_arima = r2_score(test_data[target], arima_preds)
    
    # XGBoost Metrics
    mse_xgb = mean_squared_error(test_data[target], xgb_preds)
    mae_xgb = mean_absolute_error(test_data[target], xgb_preds)
    r2_xgb = r2_score(test_data[target], xgb_preds)

    # --- Layout Plots ---
    tab1, tab2, tab3 = st.tabs(["📊 Stock & Technical Indicators", "🔮 Model Comparisons", "📈 Future Forecast (ARIMA)"])
    
    with tab1:
        st.write("### Interactive Stock Price History & Moving Averages")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['Close'], name='Close Price', line=dict(color='#3b82f6', width=2)))
        fig1.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['50_day_MA'], name='50-Day MA', line=dict(color='#f59e0b', width=1.5, dash='dash')))
        fig1.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.write("### Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=clean_df['Date'], y=clean_df['RSI'], name='RSI', line=dict(color='#ec4899', width=1.5)))
        # Overbought and Oversold lines
        fig_rsi.add_shape(type="line", x0=clean_df['Date'].iloc[0], y0=70, x1=clean_df['Date'].iloc[-1], y1=70, line=dict(color="red", width=1, dash="dash"))
        fig_rsi.add_shape(type="line", x0=clean_df['Date'].iloc[0], y0=30, x1=clean_df['Date'].iloc[-1], y1=30, line=dict(color="green", width=1, dash="dash"))
        fig_rsi.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155', range=[0, 100])
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab2:
        st.write("### Model Testing Predictions Comparison")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=test_data['Date'], y=test_data[target], name='Actual Price', line=dict(color='#ffffff', width=2)))
        fig2.add_trace(go.Scatter(x=test_data['Date'], y=arima_preds, name='ARIMA Predicted', line=dict(color='#ef4444', width=1.5)))
        fig2.add_trace(go.Scatter(x=test_data['Date'], y=xgb_preds, name='XGBoost Predicted', line=dict(color='#10b981', width=1.5)))
        fig2.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Display Accuracy Table
        st.write("### Model Performance Evaluation (On Test Dataset)")
        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score (Coefficient of Determination)"],
            "ARIMA Model": [f"{mse_arima:.4f}", f"{mae_arima:.4f}", f"{r2_arima:.4f}"],
            "XGBoost Regressor": [f"{mse_xgb:.4f}", f"{mae_xgb:.4f}", f"{r2_xgb:.4f}"]
        })
        st.table(metrics_df)

    with tab3:
        st.write("### Out-of-Sample 30-Day Future Forecast (ARIMA)")
        
        # Forecast 30 days into the future
        try:
            full_arima_model = ARIMA(clean_df[target], order=(5, 1, 1))
            full_arima_fit = full_arima_model.fit()
            future_forecast = full_arima_fit.forecast(steps=30).values
            
            last_date = clean_df['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            
            fig3 = go.Figure()
            # Show last 100 days of actual data
            actual_recent = clean_df.iloc[-100:]
            fig3.add_trace(go.Scatter(x=actual_recent['Date'], y=actual_recent['Close'], name='Recent Actual Price', line=dict(color='#3b82f6', width=2)))
            fig3.add_trace(go.Scatter(x=future_dates, y=future_forecast, name='Future Forecast', line=dict(color='#8b5cf6', width=2, dash='dot')))
            
            fig3.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#334155'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Forecast Table
            st.write("#### Detailed 30-Day Future Prediction Data")
            forecast_df = pd.DataFrame({
                "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
                "Forecasted Price ($)": [f"{p:.2f}" for p in future_forecast]
            })
            st.dataframe(forecast_df, use_container_width=True)
        except Exception as e:
            st.error(f"Future forecast generation failed: {e}")

else:
    st.info("Please select a data source to begin.")
