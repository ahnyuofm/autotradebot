import os
import json
import requests
import pyupbit
import pandas as pd
import ta
from ta.utils import dropna
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import base64
import traceback
import sqlite3

from config import (
    TRADING_PAIR,
    TRADE_FRACTION,
    MIN_TRADE_AMOUNT,
    POSITION_SIZE_PERCENTAGE
)
# Load environment variables and configurations
load_dotenv()

# Configuration
TRADING_PAIR = "KRW-ETH"
TRADE_FRACTION = 0.2
MIN_TRADE_AMOUNT = 5000

# API keys
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 로컬용
# def setup_chrome_options():
#     chrome_options = Options()
#     chrome_options.add_argument("--start-maximized")
#     chrome_options.add_argument("--headless")  # 디버깅을 위해 헤드리스 모드 비활성화
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
#     return chrome_options

# def create_driver():
#     logger.info("ChromeDriver 설정 중...")
#     service = Service(ChromeDriverManager().install())
#     driver = webdriver.Chrome(service=service, options=setup_chrome_options())
#     return driver

# EC2 서버용
def create_driver():
    logger.info("ChromeDriver 설정 중...")
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 헤드리스 모드 사용
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")

        service = Service('/usr/bin/chromedriver')  # Specify the path to the ChromeDriver executable

        # Initialize the WebDriver with the specified options
        driver = webdriver.Chrome(service=service, options=chrome_options)

        return driver
    except Exception as e:
        logger.error(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        # 요소가 뷰포트에 보일 때까지 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        # 요소가 클릭 가능할 때까지 대기
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        logger.info(f"{element_name} 클릭 완료")
        time.sleep(2)  # 클릭 후 잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 오류 발생: {e}")

def setup_database():
    conn = sqlite3.connect('trade_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (timestamp TEXT,
                  decision TEXT,
                  percentage REAL,
                  reason TEXT,
                  ETH_balance REAL,
                  krw_balance REAL,
                  eth_average_buy_price REAL,
                  eth_krw_price REAL)''')
    conn.commit()
    conn.close()

def record_trade(decision, percentage, reason):
    conn = sqlite3.connect('trade_history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    ETH_balance = upbit.get_balance("KRW-ETH")
    krw_balance = upbit.get_balance("KRW")
    eth_average_buy_price = upbit.get_avg_buy_price("KRW-ETH")
    eth_krw_price = pyupbit.get_current_price("KRW-ETH")
    
    c.execute("INSERT INTO trades (timestamp, decision, percentage, reason, ETH_balance, krw_balance, eth_average_buy_price, eth_krw_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (timestamp, decision, percentage, reason, ETH_balance, krw_balance, eth_average_buy_price, eth_krw_price))
    conn.commit()
    conn.close()

def get_trade_history(limit=10):
    conn = sqlite3.connect('trade_history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,))
    trades = c.fetchall()
    conn.close()
    return trades

def get_chart_data(symbol=TRADING_PAIR):
    def add_indicators(df):
        df = dropna(df)
        df['SMA_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['EMA_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['RSI'] = RSIIndicator(close=df['close']).rsi()
        bollinger = BollingerBands(close=df['close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        return df

    df_30d = pyupbit.get_ohlcv(symbol, interval="day", count=30)
    df_24h = pyupbit.get_ohlcv(symbol, interval="minute60", count=24)
    
    df_30d = add_indicators(df_30d)
    df_24h = add_indicators(df_24h)
    
    return {
        '30d': df_30d,
        '24h': df_24h
    }

def get_fear_and_greed_index(limit=7):
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        fng_data = [
            {
                'value': int(item['value']),
                'classification': item['value_classification'],
                'timestamp': datetime.fromtimestamp(int(item['timestamp']))
            }
            for item in data['data']
        ]
        return fng_data
    except requests.RequestException as e:
        print(f"Error fetching Fear and Greed Index: {e}")
        return None

def get_eth_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        all_articles = newsapi.get_everything(q='ethereum',
                                              language='en',
                                              sort_by='publishedAt',
                                              page_size=5)
        if all_articles['status'] == 'ok':
            articles = all_articles['articles']
            return [{"title": article["title"], "description": article["description"]} for article in articles]
        else:
            print(f"Error in API response: {all_articles['status']}")
            return []
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def take_upbit_eth_screenshot():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)

    screenshot_path = ""
    try:
        driver.get("https://upbit.com/exchange?code=CRIX.UPBIT.KRW-ETH")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "exchangeChartiq"))
        )
        time.sleep(5)
        screenshot_path = os.path.join(os.getcwd(), "upbit_eth_krw_chart.png")
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to: {screenshot_path}")
    except Exception as e:
        print(f"An error occurred during screenshot: {e}")
        driver.save_screenshot("error_state.png")
        print("Error state screenshot saved as error_state.png")
    finally:
        driver.quit()
    return screenshot_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_screenshot(image_path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant specialized in analyzing cryptocurrency trading charts. Focus on identifying trends, patterns, and potential trading signals in the Ethereum/KRW chart."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this Ethereum/KRW chart from Upbit. Provide insights on the current trend, any notable patterns, and potential trading signals. Be concise and focus on actionable information for trading decisions."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

import json
import pandas as pd

import json
import re
from openai import OpenAI
from typing import Dict, Any, Union, List
import logging
import pandas as pd
from pandas import Timestamp

logger = logging.getLogger(__name__)

def convert_to_serializable(data: Any) -> Any:
    """Convert input data to a JSON-serializable format."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    elif isinstance(data, (list, tuple)):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {str(k): convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, Timestamp):
        return data.isoformat()
    elif pd.isna(data):
        return None
    else:
        return data

def extract_decision_from_text(text: str) -> Dict[str, Any]:
    """Extract decision information from text if JSON parsing fails."""
    decision = re.search(r'\b(buy|sell|hold)\b', text.lower())
    decision = decision.group() if decision else "hold"
    
    confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    confidence = float(confidence_match.group(1)) if confidence_match else 0
    
    return {
        "decision": decision,
        "reason": text[:500] + "..." if len(text) > 500 else text,  # Truncate long texts
        "confidence": confidence,
        "analysis": {
            "market_trend": "Extracted from non-JSON response",
            "news_sentiment": "Extracted from non-JSON response",
            "technical_indicators": "Extracted from non-JSON response"
        }
    }

def get_ai_decision(market_data: Union[pd.DataFrame, List, Dict], 
                    news_data: Union[pd.DataFrame, List, Dict], 
                    screenshot_analysis: str) -> Dict[str, Any]:
    """Get trading decision from AI based on provided data."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Convert inputs to serializable format
        market_data_serializable = convert_to_serializable(market_data)
        news_data_serializable = convert_to_serializable(news_data)
        
        # Prepare the data for the AI
        serializable_data = {
            "market_data": market_data_serializable,
            "news_data": news_data_serializable,
            "screenshot_analysis": screenshot_analysis
        }
        
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "system",
                    "content": "You are an Ethereum Trading expert. Analyze the provided json data and decide whether to buy, sell, or hold Ethereum. Consider the market data, news sentiment, and screenshot analysis in your decision. Provide your decision along with a confidence level (0-1) and detailed analysis. Your response must be in valid JSON format."
                },
                {
                    "role": "user",
                    "content": json.dumps(serializable_data, default=str)
                }
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the AI's response
        print("Raw AI response:", response.choices[0].message.content)
        ai_response = response.choices[0].message.content
        logger.info(f"AI response: {ai_response}")
        
        try:
            decision_data = json.loads(ai_response)
            return decision_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.info("Attempting to extract decision from text response")
            return extract_decision_from_text(ai_response)
        
    except Exception as e:
        logger.error(f"Error in get_ai_decision: {str(e)}")
        return {
            "decision": "hold",
            "reason": f"Error in AI decision-making process: {str(e)}",
            "confidence": 0,
            "analysis": {
                "market_trend": "Error occurred",
                "news_sentiment": "Error occurred",
                "technical_indicators": "Error occurred"
            }
        }
        
def execute_trade(upbit, percentage, reason):
    try:
        current_price = pyupbit.get_current_price(TRADING_PAIR)
        
        if percentage > 0:  # Buy
            account_balance = upbit.get_balance("KRW")
            trade_amount = min(account_balance * (percentage / 100), account_balance * 0.9995)
            if trade_amount > MIN_TRADE_AMOUNT:
                order = upbit.buy_market_order(TRADING_PAIR, trade_amount)
                print(f"Buy order executed: {order}")
                print(f"Buy reason: {reason}")
            else:
                print(f"Buy failed: Trade amount ({trade_amount:.2f} KRW) is below minimum ({MIN_TRADE_AMOUNT} KRW)")

        elif percentage < 0:  # Sell
            eth_balance = upbit.get_balance(TRADING_PAIR)
            trade_amount = eth_balance * (abs(percentage) / 100)
            if trade_amount * current_price > MIN_TRADE_AMOUNT:
                order = upbit.sell_market_order(TRADING_PAIR, trade_amount)
                print(f"Sell order executed: {order}")
                print(f"Sell reason: {reason}")
            else:
                print(f"Sell failed: Trade amount ({trade_amount * current_price:.2f} KRW) is below minimum ({MIN_TRADE_AMOUNT} KRW)")

        else:  # Hold
            print(f"Holding. Reason: {reason}")

    except Exception as e:
        print(f"Error in execute_trade: {str(e)}")


def ai_trading():
    try:
        print(f"\nStarting AI trading cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        
        # Get chart data
        df = get_chart_data()
        if '30d' not in df:
            raise ValueError("Expected '30d' key in chart data not found")
        print("Chart data retrieved successfully.")
        
        # Get news
        news = get_eth_news()
        if news:
            print(f"Retrieved {len(news)} news articles.")
        else:
            print("No recent news available.")
        
        # Take and analyze screenshot
        screenshot_path = take_upbit_eth_screenshot()
        if screenshot_path and os.path.exists(screenshot_path):
            screenshot_analysis = analyze_screenshot(screenshot_path)
            print("Screenshot captured and analyzed successfully.")
        else:
            screenshot_analysis = "Failed to capture or analyze the screenshot."
            print("Warning: " + screenshot_analysis)

        # Print detailed information
        print("\nRecent price data and indicators (last 5 rows):")
        print(df['30d'].tail().to_string())
        
        if news:
            print("\nRecent Ethereum News Headlines:")
            for i, article in enumerate(news, 1):
                print(f"{i}. {article['title']}")
        
        print("\nScreenshot Analysis:")
        print(screenshot_analysis)

        # Get AI decision
        print("\nRequesting AI decision...")
        result = get_ai_decision(df['30d'], news, screenshot_analysis)
        
        # Validate and print AI decision
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict from get_ai_decision, got {type(result)}")
        
        percentage = result.get('percentage', 0)
        reason = result.get('reason', 'No reason provided')
        confidence = result.get('confidence', 0)
        
        print(f"\nAI Decision Percentage: {percentage}%")
        print(f"Confidence: {confidence}")
        print(f"Reason: {reason}")

        # Execute trade
        print("\nExecuting trade...")
        upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
        execute_trade(upbit, percentage, reason)

        print("\nAI trading cycle completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred in ai_trading: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
    finally:
        print(f"\nAI trading cycle ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    print("\nStarting live trading bot...")
    setup_database()
    while True:
        try:
            ai_trading()
            time.sleep(3600)  # Run every hour
        except KeyboardInterrupt:
            print("Bot stopped by user. Exiting gracefully...")
            break
        except Exception as e:
            print(f"An error occurred in main loop: {str(e)}")
            time.sleep(60)  # Wait a bit before retrying

if __name__ == "__main__":
    main()
