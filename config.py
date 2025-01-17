import os

# config.py

# Trading pair
TRADING_PAIR = "KRW-ETH"

# Trading parameters
TRADE_FRACTION = 0.2
MIN_TRADE_AMOUNT = 5000


# Risk management parameters
MAX_DAILY_LOSS_PERCENTAGE = 0.1  # 10% maximum daily loss
STOP_LOSS_PERCENTAGE = 0.1  # 10% stop loss

# API endpoints
ETHERSCAN_API_URL = "https://api.etherscan.io/api"
SANTIMENT_API_URL = "https://api.santiment.net/graphql"
NEWS_API_URL = "https://newsapi.org/v2/everything"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"
EETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Time intervals
TRADING_INTERVAL = 3600  # 1 hour in seconds
CHART_DATA_INTERVAL = "day"
CHART_DATA_COUNT = 30

# Configuration
TRADING_PAIR = "KRW-ETH"
TRADE_FRACTION = 0.2
MIN_TRADE_AMOUNT = 5000
STOP_LOSS_PERCENTAGE = 0.1  # 10% stop loss
MAX_DAILY_LOSS_PERCENTAGE = 0.1  # 10% maximum daily loss
POSITION_SIZE_PERCENTAGE = 0.1  # 10% of account balance per trade