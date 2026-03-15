# WSB Analyst

WSB Analyst is a full-stack application that analyzes sentiment and trends from r/wallstreetbets. The system scrapes Reddit posts, performs financial sentiment analysis using a fine-tuned FinBERT model, and generates analytics on trending tickers and simulated trading strategies.

An AI chatbot powered by Mistral 7B Instruct is also included to imitate typical WallStreetBets-style discussions.

## Tech Stack

* **Frontend:** React, TypeScript
* **Backend:** Python, Flask
* **ML:** FinBERT, Mistral 7B Instruct

## Features

* Reddit scraping pipeline for r/wallstreetbets
* Financial sentiment analysis
* Trending ticker detection and analytics
* Trade simulation based on sentiment signals
* Performance visualizations
* AI chatbot trained to imitate WSB-style users

## Running the Project

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Disclaimer

This project is for educational and research purposes only and does not constitute financial advice.
