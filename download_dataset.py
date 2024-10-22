import yfinance as yf
import pandas as pd

# Download AAPL data using yf.download()
data = yf.download("AAPL", period="max", interval="1d")

# Rename 'Date' column to 'date'
data.reset_index(inplace=True)
data.rename(columns={'Date': 'date'}, inplace=True)

# Swap 'Volume' and 'Adj Close' columns
cols = list(data.columns)
vol_index = cols.index('Volume')
adj_close_index = cols.index('Adj Close')
cols[vol_index], cols[adj_close_index] = cols[adj_close_index], cols[vol_index]
data = data[cols]

# Save the data to a CSV file
csv_path = './dataset/AAPL.csv'
data.to_csv(csv_path, index=False)

print(f"AAPL data saved to {csv_path}")