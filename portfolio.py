import yfinance as yf
import pandas as pd
import json
import time
import random
import os
from datetime import datetime, timedelta, date
from collections import defaultdict

# Add retry function for yfinance downloads with future date check
def download_with_retry(ticker, start_date, end_date, max_retries=5, base_delay=60):
    """Download data with exponential backoff retry logic for rate limits."""
    # Check if end_date is in the future
    if isinstance(end_date, str):
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
    else:
        end_date_obj = end_date.date()
    today = datetime.now().date()
    retries = 0
    while retries < max_retries:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                return data
            else:
                print(f"Empty data for {ticker} on {start_date}, retrying...")
                delay = base_delay * (retries) + random.uniform(0, 1)
                print(f"Rate limit hit for {ticker}. Retrying in {delay:.2f} seconds (attempt {retries+1}/{max_retries})")
                time.sleep(delay)
                retries += 1
        except Exception as e:
            if "Rate limit" in str(e):
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * retries + random.uniform(0, 1)
                print(f"Rate limit hit for {ticker}. Retrying in {delay:.2f} seconds (attempt {retries+1}/{max_retries})")
                time.sleep(delay)
                retries += 1
                continue
            else:
                print(f"Error downloading {ticker}: {e}")
                return pd.DataFrame()  # Return empty DataFrame for non-rate-limit errors
    
    print(f"Max retries reached for {ticker}")
    return pd.DataFrame()

# Ensure output directory exists
output_dir = "frontend/src/portfolio_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load sentiment data
with open('agg_sentiment.json', 'r') as file:
    data = json.load(file)

# Sort data by date
data = sorted(data, key=lambda x: datetime.strptime(x['day'], '%Y-%m-%d'))

def adjust_weekend_to_friday(date_str):
    """Move weekend dates to the previous Friday."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    weekday = date.weekday()
    
    if weekday == 5:  # Saturday
        return (date - timedelta(days=1)).strftime("%Y-%m-%d")  # Friday
    elif weekday == 6:  # Sunday
        return (date - timedelta(days=2)).strftime("%Y-%m-%d")  # Friday
    else:
        return date_str  # Not a weekend, return as is

def get_next_trading_day(date_str):
    """Get the next trading day (skip weekends)."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    # Increment day by 1
    next_day = date + timedelta(days=1)
    # Check if the next day is a weekend, if so, increment until it's a weekday
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day.strftime("%Y-%m-%d")

# Initialize portfolio and trading queue
portfolio = {'long': {}, 'short': {}}  # Separate long and short positions
cost_basis = {'long': {}, 'short': {}}  # Track individual position cost basis
trading_queue = defaultdict(list)  # Dictionary with date as key and list of trades as value

# For tracking actual cash invested
total_investment = 0.0  # Track total money invested (positive for buys, negative for short proceeds)
initial_investment_date = None  # Track when the first investment was made

# Process sentiment data and create trading queue
for entry in data:
    current_day = entry['day']
    next_trading_day = get_next_trading_day(current_day)
    
    # Add trade to the queue for execution on the next trading day
    trading_queue[next_trading_day].append({
        'ticker': entry['ticker'],
        'sentiment': entry['refined_sentiment']
    })

# Adjust dates - move weekend dates to previous Friday for portfolio valuation
all_dates = sorted(list(set([entry['day'] for entry in data]) | set(trading_queue.keys())))
valuation_dates = [adjust_weekend_to_friday(date) for date in all_dates]
valuation_dates = sorted(set(valuation_dates))  # Remove duplicates and re-sort

# For storing detailed portfolio data
daily_portfolio_data = []

portfolio_statistics = []

# Process trading days

daily_data = {
    'date': date,
    'trades': [],
    'positions': {'long': {}, 'short': {}},
    'today_profit': 0.0,
    'total_profit': 0.0,
    'total_investment': 0.0
}
for date in valuation_dates:
    if datetime.strptime(date, "%Y-%m-%d").date() >= datetime.now().date():
        continue
    print(f"Processing date: {date}")
    # Initialize daily data
    daily_data['date'] = date
    daily_data['trades'] = []
    daily_data['positions'] = {'long': {}, 'short': {}}
    daily_data['today_profit'] = 0.0
    total_investment = 0.0
    

    # Execute any pending trades for this date
    if date in trading_queue:
        for trade in trading_queue[date]:
            ticker = trade['ticker']
            sentiment = float(trade['sentiment'])  # Ensure sentiment is a float
            
            try:
                # Get the opening price for this ticker on this date using retry function
                end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
                price_data = download_with_retry(ticker, date, end_date)
                if not price_data.empty:
                    # Make sure we get a scalar value
                    if isinstance(price_data['Open'].iloc[0], (int, float)):
                        open_price = price_data['Open'].iloc[0]
                    else:
                        open_price = price_data['Open'].iloc[0].item()
                    
                    open_price = float(open_price)  # Convert to float for safety
                    
                    # Initialize ticker in portfolio if not present
                    if ticker not in portfolio['long']:
                        portfolio['long'][ticker] = 0.0
                        cost_basis['long'][ticker] = 0.0
                    
                    if ticker not in portfolio['short']:
                        portfolio['short'][ticker] = 0.0
                        cost_basis['short'][ticker] = 0.0
                    
                    # Track first investment date
                    if initial_investment_date is None:
                        initial_investment_date = date
                    
                    # Determine position based on sentiment
                    if sentiment <= 0:
                        # Short the stock for zero or negative sentiment
                        total_investment += abs(sentiment)
                        shares_to_add = -abs(sentiment) / open_price
                        action = "short"
                        portfolio_type = 'short'
                        # For shorts, we receive cash (negative cost basis)
                        trade_cost = shares_to_add * open_price
                    else:
                        # Long position for positive sentiment
                        total_investment += sentiment
                        shares_to_add = sentiment / open_price
                        action = "buy"
                        portfolio_type = 'long'
                        # For longs, we spend cash (positive cost basis)
                        trade_cost = shares_to_add * open_price
                    
                    # Update portfolio and cost basis for individual position
                    portfolio[portfolio_type][ticker] += shares_to_add
                    cost_basis[portfolio_type][ticker] += shares_to_add * open_price
                    
                    # Record the trade in daily data
                    daily_data['trades'].append({
                        'ticker': ticker,
                        'action': action,
                        'shares': abs(shares_to_add),
                        'price': open_price,
                        'cost': shares_to_add * open_price
                    })
                    
                    #print(f"Date: {date}, {action} {abs(shares_to_add):.4f} shares of {ticker} at ${open_price:.2f}")
                else:
                    print(f"No price data available for {ticker} on {date}")
            
            except Exception as e:
                print(f"Error processing {ticker} on {date}: {e}")
    
    # Calculate portfolio value at the end of this day
    long_profit = 0.0
    short_profit = 0.0
    # Calculate for long positions
    for ticker, shares in list(portfolio['long'].items()):
        if shares == 0:
            continue  # Skip tickers with zero shares
            
        try:
            end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
            price_data = download_with_retry(ticker, date, end_date)
            
            if not price_data.empty:
                # Ensure close_price is a scalar value
                if isinstance(price_data['Close'].iloc[0], (int, float)):
                    close_price = price_data['Close'].iloc[0]
                else:
                    close_price = price_data['Close'].iloc[0].item()
                
                close_price = float(close_price)
                
                # Calculate market value of this position
                position_value = shares * close_price # money rn
                position_cost = cost_basis['long'][ticker] # how much it costed
                cost_basis['long'][ticker] = position_value

                # Calculate P&L - for long positions, profit is current value minus cost
                position_pnl = position_value - position_cost
                long_profit += position_pnl
                
                # Record position details in daily data
                daily_data['positions']['long'][ticker] = {
                    'shares': shares,
                    'close_price': close_price,
                    'position_cost': position_cost,
                    'position_pnl': position_pnl
                }
                
                #print(f"{ticker}: {abs(shares):.4f} shares Long at ${close_price:.2f} = ${position_value:.2f}, P&L: ${position_pnl:.2f}")
        except Exception as e:
            print(f"Error valuing {ticker} on {date}: {e}")
    
    # Calculate for short positions
    for ticker, shares in list(portfolio['short'].items()):
        if shares == 0:
            continue  # Skip tickers with zero shares
        
        try:
            end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
            price_data = download_with_retry(ticker, date, end_date)
            
            if not price_data.empty:
                # Ensure close_price is a scalar value
                if isinstance(price_data['Close'].iloc[0], (int, float)):
                    close_price = price_data['Close'].iloc[0]
                else:
                    close_price = price_data['Close'].iloc[0].item()
                
                close_price = float(close_price)
                
                # Calculate market value of this position (negative for shorts)
                position_value = shares * close_price  # current price
                position_cost = cost_basis['short'][ticker]  #  price u shorted at
                cost_basis['short'][ticker] = position_value

                # Calculate P&L - for short positions, profit is cost minus current value
                position_pnl = position_cost - position_value
                short_profit += position_pnl
                
                # Record position details in daily data
                daily_data['positions']['short'][ticker] = {
                    'shares': shares,
                    'close_price': close_price,
                    'position_value': position_value,
                    'position_cost': position_cost,
                    'position_pnl': position_pnl
                }
                
                #print(f"{ticker}: {abs(shares):.4f} shares Short at ${close_price:.2f} = ${position_value:.2f}, P&L: ${position_pnl:.2f}")
        except Exception as e:
            print(f"Error valuing {ticker} on {date}: {e}")
    
    
    # Update portfolio value and total investment in daily data
    daily_data['today_profit'] = long_profit + short_profit
    daily_data['total_profit'] += daily_data['today_profit']
    daily_data['total_investment'] += total_investment
    
    # Add to history collections
    daily_portfolio_data.append(daily_data)
    

    portfolio_statistics.append({
        'date': date,
        'investment': daily_data['total_investment'],
        'today_profit': daily_data['today_profit'],
        'total_profit': daily_data['total_profit']
    })
    
    # Save daily data to JSON file
    filename = os.path.join(output_dir, f"portfolio_{date}.json")
    with open(filename, 'w') as file:
        json.dump(daily_data, file, indent=2)
    
    print(f"Total investment: ${daily_data['total_investment']:.2f}")
    print(f"Daily P&L on {date}: ${daily_data['today_profit']:.2f}")
    print(f"Overall Profit: ${daily_data['total_profit']:.2f}")
    print(f"Saved daily data to {filename}")
    print("-" * 50)

# Save the complete history to a single file
history_filename = os.path.join(output_dir, "portfolio_total_investment.json")
with open(history_filename, 'w') as file:
    json.dump({
        'daily_data': daily_portfolio_data,
        'portfolio_statistics': portfolio_statistics,
        'initial_investment_date': initial_investment_date,
        'total_investment': daily_data['total_investment']
    }, file, indent=2)
print(f"Saved complete history to {history_filename}")

# Output final results
print("\nFinal Portfolio:")
print("\nLong Positions:")
for ticker, shares in portfolio['long'].items():
    if shares != 0:  # Show all non-zero positions
        print(f"{ticker}: {abs(shares):.4f} shares Buy, Cost Basis: ${cost_basis['long'][ticker]:.2f}")

print("\nShort Positions:")
for ticker, shares in portfolio['short'].items():
    if shares != 0:  # Show all non-zero positions
        print(f"{ticker}: {abs(shares):.4f} shares Short, Cost Basis: ${cost_basis['short'][ticker]:.2f}")


print("\nDaily P&L History:")
for record in portfolio_statistics:
    print(f"{record['date']}: ${record['today_profit']:.2f}")

# Calculate total return
if portfolio_statistics:
    total_return = sum(record['today_profit'] for record in portfolio_statistics)
    print(f"\nCumulative P&L: ${total_return:.2f}")