import json
from collections import defaultdict
from datetime import datetime

aggregated_data = defaultdict(lambda: {'sentiment_score': 0, 'upvotes': 0, 'day': ''})

with open('final_ticker_sentiments.json', 'r') as file:
    data = json.load(file)

# Process each post in the data
for entry in data:
    ticker = entry["ticker"]
    if ticker == "SPX":
        ticker = "SPY"
    created_at = entry["created_at"]
    sentiment_score = entry["sentiment_score"]
    upvotes = entry["upvotes"]

    # Extract the day (date part only)
    day = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").date()

    # Update the aggregated sentiment and upvotes for the ticker and day
    day_str = str(day)
    aggregated_data[(ticker, day_str)]['sentiment_score'] += sentiment_score * upvotes
    aggregated_data[(ticker, day_str)]['day'] = day.strftime('%Y-%m-%d')

# Prepare the final output in the desired format
final_data = []
for (ticker, day), values in aggregated_data.items():
    refined_sentiment = values['sentiment_score']
    final_data.append({
        "ticker": ticker,
        "refined_sentiment": refined_sentiment,
        "day": day
    })

# Output the result as JSON
output_json = json.dumps(final_data, indent=4)

# Save to a new JSON file (optional)
with open('agg_sentiment.json', 'w') as f:
    json.dump(final_data, f, indent=4)