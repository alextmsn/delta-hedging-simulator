# 💹 Delta Hedging Simulator

This interactive Streamlit app lets you simulate the process of delta hedging an option position using real or uploaded financial data. It includes full Black-Scholes valuation, Greeks calculation, hedge error tracking, and dynamic charting to help you understand the mechanics and trade-offs in delta-neutral strategies.

---

## ⚙️ Features

- 🧮 **Black-Scholes Option Pricing**
  - Uses `py_vollib` if available, falls back to custom implementation otherwise
  - Computes delta, gamma, theta, and vega on each step

- 📈 **Real-Time or Custom Market Data**
  - Pulls historical stock data via Yahoo Finance
  - Option to upload your own CSV with price series

- 🔁 **Configurable Simulation**
  - Choose call/put, long/short position
  - Set strike price, time to maturity, implied volatility
  - Define hedge frequency and transaction cost

- 📊 **Live Charts**
  - Candlestick price chart with cumulative hedge error overlay
  - Option value and delta visualized over time
  - Fully interactive: zoom, pan, isolate with legend clicks

- 📤 **Exportable Results**
  - Outputs a CSV file containing price path, option value, delta, hedge error, and P&L series

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/alextmsn/delta-hedging-simulator.git
cd delta-hedging-simulator

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py


AUTHOUR
Made by @alextmsn
MIT Licence - 2025

