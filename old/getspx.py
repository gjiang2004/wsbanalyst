import yfinance as yf
import pandas as pd
import json

# Define the ticker symbol for the S&P 500 index
ticker_symbol = '^GSPC'

# Define the start and end dates for the data retrieval
start_date = '2025-02-28'
end_date = '2025-03-30'

# Fetch the historical data for the specified date range
spx_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Flatten multi-index columns (if present)
spx_data.columns = [col[0] if isinstance(col, tuple) else col for col in spx_data.columns]

# Get the opening price of the first day
start = spx_data['Open'].iloc[0]

# Calculate the percent change from the starting price
spx_data['Percent Change'] = (spx_data['Close'] - start) / start

# Drop NaN values
spx_data = spx_data.dropna()

# Reset index to convert dates into a normal column
spx_data = spx_data.reset_index()

# Convert the 'Date' column to string to avoid JSON serialization issues
spx_data['Date'] = spx_data['Date'].astype(str)

# Drop the unwanted columns
spx_data = spx_data[['Date', 'Percent Change']]

# Convert DataFrame to a dictionary with records format
spx_data_json = spx_data.to_dict(orient='records')

# Save to a formatted JSON file
with open("spx_data.json", "w") as json_file:
    json.dump(spx_data_json, json_file, indent=4)

