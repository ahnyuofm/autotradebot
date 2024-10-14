import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with file and console output"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create handlers
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  # 10MB file size
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Create main logger
main_logger = setup_logger('main', f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log')

# Create separate logger for trade executions
trade_logger = setup_logger('trades', 'logs/trade_executions.log')

def log_trade(action, amount, price, reason):
    """Function to log trade executions"""
    trade_logger.info(f"Trade executed - Action: {action}, Amount: {amount}, Price: {price}, Reason: {reason}")

# Example usage
if __name__ == "__main__":
    main_logger.info("This is a test log message")
    log_trade("BUY", 0.1, 3000, "RSI oversold")