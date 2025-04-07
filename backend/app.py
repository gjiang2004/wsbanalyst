# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import json
import re

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Replace with your actual API key
API_KEY = "AIzaSyAJZBZN5HBWXtM5zPZiKOyee_MssBU7Htw"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Files for conversation history
BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base.txt")
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.txt")
SENTIMENT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agg_sentiment.json")

def load_history():
    if os.path.exists(BASE_FILE):
        with open(BASE_FILE, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        file.write(history)

# Load conversation history on startup
conversation_history = load_history()

# Load sentiment data
try:
    with open(SENTIMENT_FILE, "r") as f:
        sentiment_data = json.load(f)
except Exception as e:
    print(f"Error loading sentiment data: {e}")
    sentiment_data = []

@app.route("/", methods=["GET"])
def index():
    return "Backend is running!", 200

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history

    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Sort sentiment data by descending refined_sentiment
    sorted_tickers = sorted(sentiment_data, key=lambda x: x['refined_sentiment'], reverse=True)
    top_5 = sorted_tickers[:5]
    # Build a small list of top 5
    ticker_summary = ", ".join(f"{item['ticker']}" for item in top_5)

    # Prompt instructing the model to avoid bracket/markdown formatting
    prompt = f"""
Act like a normal WSB user. Keep responses somewhat short and do not use bracket or markdown formatting.
Here are the top 5 tickers based on sentiment that you can use for reference: {ticker_summary} Make conversation with the user. Try not to repeat yourself.
Use thought provoking questions about the market, think of ways to cope hard, make sure to include WSB terminology.
Think about earning money, when you lose cope hard, when you win HOLD TO THE MOON
If the user asks for a specific stock refer to the top 5 tickers based on sentiment.
Talk about investment ideas with the user.
Do not use bad words or language that people find offensive.

User: {user_input}

Conversation history:
{conversation_history}
"""

    response = model.generate_content(prompt)
    bot_response = response.text.strip()
    bot_response = re.sub(r"\[.*?\]", "", bot_response)


    # If you want to absolutely remove leftover bracket text, see optional Step 4 below.

    # Update conversation history
    conversation_history = user_input + "\n" + bot_response + "\n" + conversation_history
    save_history(conversation_history)

    return jsonify({"response": bot_response})

@app.route("/history", methods=["GET"])
def get_history():
    history = load_history()
    return jsonify({"history": history})

if __name__ == "__main__":
    app.run(debug=True)