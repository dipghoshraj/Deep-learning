import yfinance as yf
import pandas as pd
from datetime import datetime

# Set the time range
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - 10)).strftime('%Y-%m-%d')

# Define tickers
tickers = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Crude_Oil': 'CL=F',
    'USD_INR': 'INR=X',
    'DXY': 'DX-Y.NYB',
    'EUR_USD': 'EURUSD=X',
    'S&P500': '^GSPC',
    'Nasdaq': '^IXIC',
    'Nikkei': '^N225',
    'FTSE100': '^FTSE',
    'VIX': '^VIX'
}

# Download data
for name, symbol in tickers.items():
    print(f"Downloading {name} ({symbol})...")
    data = yf.download(symbol, start=start_date, end=end_date)
    data.to_csv(f"{name}_10yr_data.csv")

print("✅ All available data has been downloaded.")
