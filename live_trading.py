import time
from pyupbit import Upbit
from config import TRADING_PAIR, TRADING_INTERVAL
from os
import json
import requests
import pandas as pd
import ta
import time
from dotenv import load_dotenv
from openai import OpenAI

def live_trading_loop():
    upbit = Upbit(access_key, secret_key)
    
    access_key = os.getenv('UPBIT_ACCESS_KEY')
    secret_key = os.getenv('UPBIT_SECRET_KEY')
    
    while True:
        try:
            # Fetch current market data
            df = get_chart_data()
            eth_sentiment_index = get_eth_custom_index()
            news = get_eth_news()

            # Get AI decision
            decision = get_ai_decision(df, eth_sentiment_index, news)

            print(f"AI Decision: {decision['decision']}")
            print(f"Reason: {decision['reason']}")

            # Execute trade based on decision
            execute_trade(upbit, decision['decision'], decision['reason'])

            # Wait for next trading interval
            time.sleep(TRADING_INTERVAL)

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    print("Starting live trading...")
    live_trading_loop()