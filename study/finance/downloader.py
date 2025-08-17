# =============================================================================
# DISCLAIMER
# =============================================================================
# For getting a public version of the signal software, please visit:
#   👉 git@github.com:dipghoshraj/Deep-learning.git
#
# This notebook is a public version of the code and represents a **base version**
# of the actual product.
#
# For access to:
#   ✅ Full product
#   ✅ Advanced features
#   ✅ Complete signal software
# Please contact:
#   📧 dipghoshraj@gmail.com
# =============================================================================



import yfinance as yf
import pandas as pd
from datetime import datetime

# Range of year

year_range = 15


# Date range
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - year_range)).strftime('%Y-%m-%d')

# Tickers and readable names
tickers = {
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'CL=F': 'Crude_Oil',
    'INR=X': 'USD_INR',
    'DX-Y.NYB': 'DXY',
    'EURUSD=X': 'EUR_USD',
    '^GSPC': 'S&P500',
    '^IXIC': 'Nasdaq',
    '^N225': 'Nikkei',
    '^FTSE': 'FTSE100',
    '^VIX': 'CBOE_VIX',
    '^NSEI': 'Nifty_50',
}

# Loop through tickers
for symbol, name in tickers.items():
    print(f"📥 Downloading {name} ({symbol})...")
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        print(f"⚠️ No data for {name} ({symbol}). Skipping.")
        continue

    # Drop Adj Close if it's not needed
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True)

    # Reset index (Date) into a column
    df.reset_index(inplace=True)

    # Reorder and clean columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Save each asset to its own CSV
    filename = f"storage/{name}_{year_range}yr_data.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved to: {filename}")

print("✅ All data downloaded and saved in separate clean files.")
