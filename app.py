import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from core.hedging import hedge_option_path
import yfinance as yf
import ast
import io
from scipy.stats import norm

# Fallback Black-Scholes implementation if py_vollib is unavailable
try:
    from py_vollib.black_scholes import black_scholes as bs
    from py_vollib.black_scholes.greeks.analytical import delta as delta_bs, gamma, theta, vega
    PY_VOLLIB_AVAILABLE = True
except ImportError:
    PY_VOLLIB_AVAILABLE = False
    st.warning("py_vollib not installed. Using simplified Black-Scholes implementation. Install with: pip install py-vollib")

    def black_scholes(flag, S, K, t, r, sigma):
        """Simplified Black-Scholes for call/put pricing."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if flag == 'c':
            price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    def delta_bs(flag, S, K, t, r, sigma):
        """Simplified delta calculation."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        if flag == 'c':
            return norm.cdf(d1)
        return norm.cdf(d1) - 1

    def gamma(flag, S, K, t, r, sigma):
        """Simplified gamma calculation."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        return norm.pdf(d1) / (S * sigma * np.sqrt(t))

    def theta(flag, S, K, t, r, sigma):
        """Simplified theta calculation (approximate)."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        if flag == 'c':
            return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
        return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)

    def vega(flag, S, K, t, r, sigma):
        """Simplified vega calculation."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        return S * norm.pdf(d1) * np.sqrt(t)

# Set page configuration
st.set_page_config(page_title="Delta Hedging Simulator", layout="wide")

# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2d2d2d;
    }
    .stExpander {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("💹 Delta Hedging Simulator")
st.subheader("Advanced Options Trading Analysis")

# Sidebar with expanders
with st.sidebar:
    st.markdown("### Settings")
    with st.expander("⚙️ Option Settings", expanded=True):
        st.info("Select option type and position for the simulation.")
        option_type = st.selectbox("Option Type", ["call", "put"])
        position = st.selectbox("Position", ["short", "long"])
        K = st.number_input("Strike Price", 50.0, 1000.0, 200.0, help="Price at which the option can be exercised.")
        T = st.number_input("Time to Maturity (years)", 0.01, 2.0, 0.5, step=0.01, help="Time until option expiration.")

    with st.expander("📉 Volatility Settings"):
        st.info("Set the implied volatility, typically 20-50% for stocks.")
        sigma = st.slider("Implied Vol (%)", 1, 200, 50) / 100

    with st.expander("🛡️ Hedging Settings"):
        st.info("Configure hedging frequency and transaction costs.")
        hedge_freq = st.slider("Hedge Every N Steps", 1, 50, 1)
        transaction_cost = st.slider("Transaction Cost (%)", 0.0, 5.0, 0.0) / 100

    with st.expander("📈 Load Market Data"):
        st.info("Choose data source: Yahoo Finance for live data or upload CSV. Use '1d' interval for near-real-time data.")
        data_source = st.radio("Select Source", ["Yahoo Finance", "Upload CSV"])
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Enter Ticker Symbol", value="AAPL", help="E.g., AAPL for Apple Inc.")
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"])
            interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"])
        else:
            uploaded = st.file_uploader("Upload CSV with a price column (e.g., 'Close', 'Price')", type="csv", help="CSV must have a price column; 'Open', 'High', 'Low' optional for candlestick chart.")
            if uploaded:
                df_temp = pd.read_csv(uploaded)
                df_temp.columns = [col[0] if isinstance(col, tuple) else col.strip().title() for col in df_temp.columns]
                price_cols = [col for col in df_temp.columns if col.lower() in ['close', 'price', 'adj close', 'adjusted close', 'stockprice', 'value']]
                if price_cols:
                    default_price_col = price_cols[0]
                else:
                    default_price_col = df_temp.columns[0] if df_temp.columns.size > 0 else ""
                selected_price_col = st.selectbox("Select Price Column", df_temp.columns, index=df_temp.columns.get_loc(default_price_col) if default_price_col in df_temp.columns else 0, help="Choose the column containing price data.")

    # Dark mode toggle
    if st.button("Toggle Dark Mode"):
        st.session_state.dark_mode = not getattr(st.session_state, 'dark_mode', False)
        if st.session_state.dark_mode:
            st.markdown("""
                <style>
                .stApp { background-color: #1a1a1a; color: #ffffff; }
                .stSidebar { background-color: #2d2d2d; }
                .stExpander { background-color: #2d2d2d; color: #ffffff; }
                .stPlotlyChart { color: #ffffff; }
                </style>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <style>
                .stApp { background-color: white; color: black; }
                .stSidebar { background-color: #f0f0f0; }
                .stExpander { background-color: #f0f0f0; color: black; }
                .stPlotlyChart { color: black; }
                </style>
                """, unsafe_allow_html=True)

    run = st.button("🚀 Run Simulation")

# Main execution
if run:
    with st.spinner("Fetching market data..."):
        if data_source == "Yahoo Finance":
            try:
                df = yf.download(ticker, period=period, interval=interval)
                if df.empty:
                    st.error("No data found for the given ticker.")
                    st.stop()
            except Exception as e:
                st.error(f"Data download failed: {e}")
                st.stop()
            # Standardize Yahoo Finance columns
            df.columns = [col[0] if isinstance(col, tuple) else col.strip().title() for col in df.columns]
            # Ensure 'Close' column exists
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'Close'})
        else:
            if not uploaded:
                st.warning("Please upload a CSV file with a price column.")
                st.stop()
            df = pd.read_csv(uploaded)
            # Standardize column names
            df.columns = [col[0] if isinstance(col, tuple) else col.strip().title() for col in df.columns]
            # Rename selected price column to 'Close'
            if selected_price_col != 'Close':
                df = df.rename(columns={selected_price_col: 'Close'})
            if 'Close' not in df.columns:
                st.error(f"'Close' column not found. Available columns: {list(df.columns)}")
                st.stop()

        # Handle missing candlestick columns early
        for col in ['Open', 'High', 'Low']:
            if col not in df.columns:
                df[col] = df['Close']  # Fallback to Close
        df = df[['Open', 'High', 'Low', 'Close']].dropna()

        # Create a date index if missing
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='D')

        # Handle potential list-like strings in CSV
        if data_source == "Upload CSV":
            def parse_value(x):
                if isinstance(x, str) and x.startswith('['):
                    try:
                        return float(ast.literal_eval(x)[0])
                    except (ValueError, SyntaxError):
                        return np.nan
                return x
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].apply(parse_value)

        # Convert columns to float
        try:
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            path = df['Close'].dropna().tolist()
        except Exception as e:
            st.error(f"Failed to parse price columns: {e}")
            st.stop()

        steps = len(path) - 1
        if steps <= 0:
            st.error("Price series is too short.")
            st.stop()
        dt = T / steps

    with st.spinner("Running simulation..."):
        # Run Delta Hedge Simulation
        try:
            pnl, error_series = hedge_option_path(
                path, K, sigma, 0.0, dt,
                option_type=option_type,
                position=position,
                hedge_every_n=hedge_freq,
                transaction_cost=transaction_cost
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

        # Compute option value and delta series
        option_value_series = []
        delta_series = []
        for i in range(len(path)):
            time_to_maturity = max(T - i * dt, 0.0001)  # Avoid T=0 for numerical stability
            S = path[i]
            try:
                ov = bs(option_type[0], S, K, time_to_maturity, 0.0, sigma)
                d = delta_bs(option_type[0], S, K, time_to_maturity, 0.0, sigma)
                if position == "short":
                    ov = -ov
                    d = -d
                option_value_series.append(ov)
                delta_series.append(d)
            except Exception as e:
                st.warning(f"Option calculation failed at step {i}: {e}")
                option_value_series.append(0.0)
                delta_series.append(0.0)

        # Compute Greeks at second-to-last step for final values
        if len(path) > 1:
            final_time = max(T - (len(path)-2) * dt, 0.0001)
            final_S = path[-2]
            try:
                delta_final = delta_bs(option_type[0], final_S, K, final_time, 0.0, sigma)
                gamma_final = gamma(option_type[0], final_S, K, final_time, 0.0, sigma)
                theta_final = theta(option_type[0], final_S, K, final_time, 0.0, sigma)
                vega_final = vega(option_type[0], final_S, K, final_time, 0.0, sigma)
                if position == "short":
                    delta_final = -delta_final
                    gamma_final = -gamma_final
                    theta_final = -theta_final
                    vega_final = -vega_final
            except Exception as e:
                st.warning(f"Greeks calculation failed: {e}")
                delta_final = gamma_final = theta_final = vega_final = 0.0
        else:
            delta_final = gamma_final = theta_final = vega_final = 0.0

        # Summary statistics
        max_drawdown = 0  # Placeholder, need P&L series for accurate calculation
        num_hedges = (steps // hedge_freq) + 1 if steps > 0 else 0
        avg_error = np.mean(error_series) if error_series else 0.0

    # Display results in cards
    st.subheader("📊 Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h3 style='color:#{'2ecc71' if pnl >= 0 else 'e74c3c'};'>Final P&L: ${pnl:.2f}</h3>", unsafe_allow_html=True)
        st.markdown("### Option Greeks")
        st.write(f"Delta: {delta_final:.4f} (Price sensitivity)")
        st.write(f"Gamma: {gamma_final:.4f} (Delta sensitivity)")
        st.write(f"Theta: {theta_final:.4f} (Time decay)")
        st.write(f"Vega: {vega_final:.4f} (Volatility sensitivity)")
    with col2:
        st.markdown("### Summary Statistics")
        st.write(f"Max Drawdown: ${max_drawdown:.2f}")
        st.write(f"Number of Hedges: {num_hedges}")
        st.write(f"Average Hedging Error: {avg_error:.4f}")

    # Tabs for plots
    tab1, tab2, tab3 = st.tabs(["Price & Error", "Option Value", "Delta"])

    with tab1:
        # Check if candlestick data is valid
        if df[['Open', 'High', 'Low', 'Close']].notnull().all().all():
            # Candlestick chart for price and line for error
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='lime',
                decreasing_line_color='red',
                whiskerwidth=1,
                line_width=3  # Thicker candle borders for visibility
            ))
            fig.add_trace(go.Scatter(x=df.index, y=error_series, mode='lines', name='Cumulative Hedge Error',
                                   line=dict(color='#e74c3c', width=3)))
            fig.update_layout(title="Underlying Asset Price History & Cumulative Hedge Error",
                             xaxis_title="Date",
                             yaxis_title="Price / Error",
                             template="plotly_dark",
                             font=dict(size=14),
                             xaxis_rangeslider_visible=False)
        else:
            # Fallback to line plot
            st.warning("Incomplete candlestick data. Displaying price as line plot.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price',
                                   line=dict(color='#2ecc71', width=3)))
            fig.add_trace(go.Scatter(x=df.index, y=error_series, mode='lines', name='Cumulative Hedge Error',
                                   line=dict(color='#e74c3c', width=3)))
            fig.update_layout(title="Underlying Asset Price & Cumulative Hedge Error",
                             xaxis_title="Date",
                             yaxis_title="Price / Error",
                             template="plotly_dark",
                             font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Tip:** Use your mouse to explore the chart. Scroll to zoom, drag to pan. Click on a legend item to hide a line. **Double-click a legend item to isolate a single line**, and double-click again to reset.")
        
        

    with tab2:
        # Option value over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=option_value_series, mode='lines', name='Option Value',
                               line=dict(color='#2ecc71', width=3)))
        fig.update_layout(title="Option Theoretical Value Over Time",
                         xaxis_title="Date",
                         yaxis_title="Option Value",
                         template="plotly_dark",
                         font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Delta over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=delta_series, mode='lines', name='Delta',
                               line=dict(color='#3498db', width=3)))
        fig.update_layout(title="Option Delta Over Time",
                         xaxis_title="Date",
                         yaxis_title="Delta",
                         template="plotly_dark",
                         font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    # Debug array lengths before creating DataFrame
    lengths = {
        'Date': len(df.index),
        'Price': len(path),
        'Option Value': len(option_value_series),
        'Delta': len(delta_series),
        'Hedge Error': len(error_series),
        'P&L': len([pnl] * len(path))
    }
    st.write("Array lengths for export:", lengths)  # Debugging output
    target_length = len(df.index)
    if not all(length == target_length for length in lengths.values()):
        st.error(f"Length mismatch detected. All arrays must have length {target_length}. Current lengths: {lengths}")
        # Adjust error_series to match target_length
        if len(error_series) < target_length:
            error_series.extend([0.0] * (target_length - len(error_series)))  # Pad with zeros
        elif len(error_series) > target_length:
            error_series = error_series[:target_length]  # Truncate
        st.warning(f"Adjusted 'Hedge Error' length to {len(error_series)} to match DataFrame index.")

    # Export functionality
    try:
        export_data = pd.DataFrame({
            'Date': df.index,
            'Price': path,
            'Option Value': option_value_series,
            'Delta': delta_series,
            'Hedge Error': error_series,
            'P&L': [pnl] * len(path)  # Placeholder, need actual P&L series
        })
        csv = export_data.to_csv(index=False)
        st.download_button(label="Export Results", data=csv, file_name='delta_hedging_results.csv',
                          mime='text/csv')
    except Exception as e:
        st.error(f"Failed to create export DataFrame: {e}")
        st.stop()

# Footer
st.markdown("---")
st.markdown("© 2025 Delta Hedging Simulator | Learn more about delta hedging at [Investopedia](https://www.investopedia.com/terms/d/deltahedging.asp)")