import os
import json
import requests
import pyupbit
import pandas as pd
import ta
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from openai import OpenAI
from config import (
    TRADING_PAIR,
    TRADE_FRACTION,
    MIN_TRADE_AMOUNT,
    STOP_LOSS_PERCENTAGE,
    MAX_DAILY_LOSS_PERCENTAGE,
    POSITION_SIZE_PERCENTAGE
)

load_dotenv()

# Configuration
TRADING_PAIR = "KRW-ETH"
TRADE_FRACTION = 0.2
MIN_TRADE_AMOUNT = 5000

# API keys
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
api_key = os.getenv('OPENAI_API_KEY')

def get_chart_data(symbol=TRADING_PAIR, count=30, interval="day"):
    df = pyupbit.get_ohlcv(symbol, count=count, interval=interval)
    
    # Add technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['close']).macd_signal()
    df['BB_High'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
    df['BB_Low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
    
    return df

#def get_eth_gas_price():
    url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={ETHERSCAN_API_KEY}"
    
    try:
        response = requests.get(url)
        
        # Check if response is valid (status code 200)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data from API. Status Code: {response.status_code}")
        
        data = response.json()
        
        # Check if the 'result' key is present in the response
       # In get_eth_custom_index(), handle when data is None
        if gas_price is None or social_sentiment is None:
            print("Error: Gas price or sentiment data missing. Cannot calculate index.")
            return None

        
        # Check if 'SafeGasPrice' is in the 'result' and convert it to a number
        safe_gas_price = data['result'].get('SafeGasPrice', None)
        if safe_gas_price is None:
            raise ValueError(f"'SafeGasPrice' is missing in the response. Full response: {data['result']}")
        
        # Convert 'SafeGasPrice' to float then to int (in case it's a float string)
        return int(float(safe_gas_price))
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Return None if an error occurs
    return None

#def get_eth_social_sentiment():
    url = "https://api.santiment.net/graphql"
    query = """
    {
      getMetric(metric: "sentiment_volume_consumed_total") {
        timeseriesData(
          slug: "ethereum"
          from: "2023-10-10T00:00:00Z"
          to: "2024-09-09T23:59:59Z"
          interval: "1d"
        ) {
          value
        }
      }
    }
    """
    headers = {"Authorization": f"Bearer {SANTIMENT_API_KEY}"}
    try:
        response = requests.post(url, json={"query": query}, headers=headers)
        data = response.json()
        
        if 'data' in data and 'getMetric' in data['data'] and 'timeseriesData' in data['data']['getMetric']:
            sentiment_values = data['data']['getMetric']['timeseriesData']
            if sentiment_values:
                return sentiment_values[-1]['value']  # Get the last value in the timeseries
        else:
            print(f"Error: Invalid response structure: {data}")
        
        return None
    except Exception as e:
        print(f"Error in get_eth_social_sentiment: {str(e)}")
        return None


#def get_eth_custom_index():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url)
        data = response.json()
        price_change = data['ethereum']['usd_24h_change']

        gas_price = get_eth_gas_price()
        social_sentiment = get_eth_social_sentiment()

        if gas_price is None or social_sentiment is None:
            print("Error: Gas price or sentiment data missing. Cannot calculate index.")
            return None

        custom_index = (
            price_change * 0.4 +
            gas_price * 0.3 +
            social_sentiment * 0.3
        )

        return max(0, min(custom_index, 100))
    except Exception as e:
        print(f"Error in get_eth_custom_index: {str(e)}")
        return None

from newsapi import NewsApiClient

# Make sure NEWS_API_KEY is set in your environment variables or .env file
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def get_eth_news():
    try:
        # Initialize the client
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        # Fetch Ethereum-related news
        all_articles = newsapi.get_everything(q='ethereum',
                                              language='en',
                                              sort_by='publishedAt',
                                              page_size=5)

        # Process the articles
        if all_articles['status'] == 'ok':
            articles = all_articles['articles']
            return [{"title": article["title"], "description": article["description"]} for article in articles]
        else:
            print(f"Error in API response: {all_articles['status']}")
            return []

    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def get_ai_decision(df, news):
    try:
        client = OpenAI()
        news_text = "\n".join([f"Title: {article['title']}\nDescription: {article['description']}" for article in news])
        
        df_dict = df.reset_index().to_dict(orient='records')
       
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                "role": "system",
                "content": """You are an Ethereum Trading expert. Analyze the provided json data and decide whether to buy, sell, or hold based on the information provided. Consider the technical indicators and the Ethereum Sentiment Index in your decision.

Your response must be a JSON object with the following structure:
{
    "decision": "buy|sell|hold",
    "reason": "A brief explanation of your decision"
}"""
                },
                {
                 "role": "user",
                "content": json.dumps({
                    "market_data": df_dict,
                    #"eth_sentiment_index": eth_sentiment_index,
                    "recent_news": news_text
                }, default=str)
                },
            ],
            response_format={"type": "json_object"}
        )
        decision_data = json.loads(response.choices[0].message.content)
        
        print(f"AI Response: {decision_data}")
        
        if 'decision' not in decision_data or 'reason' not in decision_data:
            raise ValueError(f"AI response is missing 'decision' or 'reason'. Full response: {decision_data}")
        
        return decision_data
    except Exception as e:
        print(f"Error in get_ai_decision: {str(e)}")
        return {"decision": "hold", "reason": "Error in AI decision-making process"}



def calculate_daily_pnl(upbit):
    try:
        balances = upbit.get_balances()
        total_value = sum(float(balance['balance']) * float(balance['avg_buy_price']) for balance in balances if 'balance' in balance and 'avg_buy_price' in balance)
        initial_balance = float(os.getenv("INITIAL_BALANCE", str(total_value)))
        return (total_value - initial_balance) / initial_balance
    except Exception as e:
        print(f"Error in calculate_daily_pnl: {str(e)}")
        return 0

def execute_trade(upbit, decision, reason):
    try:
        current_price = pyupbit.get_current_price(TRADING_PAIR)
        account_balance = upbit.get_balance("KRW")
        
        # Check daily loss limit
        daily_pnl = calculate_daily_pnl(upbit)
        if daily_pnl < -MAX_DAILY_LOSS_PERCENTAGE:
            print(f"Daily loss limit reached: {daily_pnl:.2%}. Skipping trade.")
            return

        if decision == "buy":
            # Position sizing
            trade_amount = min(account_balance * POSITION_SIZE_PERCENTAGE, account_balance * 0.9995)
            if trade_amount > MIN_TRADE_AMOUNT:
                order = upbit.buy_market_order(TRADING_PAIR, trade_amount)
                print(f"Buy order executed: {order}")
                print(f"Buy reason: {reason}")
                
                # Set stop loss order
                stop_loss_price = current_price * (1 - STOP_LOSS_PERCENTAGE)
                eth_balance = upbit.get_balance(TRADING_PAIR)
                upbit.sell_limit_order(TRADING_PAIR, stop_loss_price, eth_balance)
                print(f"Stop loss order set at {stop_loss_price}")
            else:
                print(f"Buy failed: Trade amount ({trade_amount:.2f} KRW) is below minimum ({MIN_TRADE_AMOUNT} KRW)")
        
        elif decision == "sell":
            eth_balance = upbit.get_balance(TRADING_PAIR)
            trade_amount = eth_balance * TRADE_FRACTION
            if trade_amount * current_price > MIN_TRADE_AMOUNT:
                order = upbit.sell_market_order(TRADING_PAIR, trade_amount)
                print(f"Sell order executed: {order}")
                print(f"Sell reason: {reason}")
            else:
                print(f"Sell failed: Trade amount ({trade_amount * current_price:.2f} KRW) is below minimum ({MIN_TRADE_AMOUNT} KRW)")
        
        elif decision == "hold":
            print(f"Holding. Reason: {reason}")

    except Exception as e:
        print(f"Error in execute_trade: {str(e)}")

# Make sure to call this function in your main trading loop
def cancel_all_orders(upbit):
    orders = upbit.get_order(TRADING_PAIR)
    for order in orders:
        upbit.cancel_order(order['uuid'])
    print("All existing orders cancelled.")

# Update your main trading loop to include this:

def cancel_all_orders(upbit):
    try:
        orders = upbit.get_order(TRADING_PAIR)
        for order in orders:
            upbit.cancel_order(order['uuid'])
        print("All existing orders cancelled.")
    except Exception as e:
        print(f"Error in cancel_all_orders: {str(e)}")

def ai_trading():
    try:
        df = get_chart_data()
        news = get_eth_news()

        print("Recent price data and indicators:")
        print(df.tail())
        
        if news:
            print("Recent Ethereum News Headlines:")
            for article in news:
                print(f"- {article['title']}")
        else:
            print("No recent news available.")

        result = get_ai_decision(df, news)
        
        upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
        execute_trade(upbit, result["decision"], result["reason"])

    except Exception as e:
        print(f"An error occurred in ai_trading: {str(e)}")


def main():
    print("\nStarting live trading bot...")
    while True:
        try:
            ai_trading()
            time.sleep(3600)
        except KeyboardInterrupt:
            print("Bot stopped by user. Exiting gracefully...")
            break
        except Exception as e:
            print(f"An error occurred in main loop: {str(e)}")
            time.sleep(60)  # Wait a bit before retrying

if __name__ == "__main__":
    main()
