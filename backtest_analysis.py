import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test import backtest
from config import TRADING_PAIR, BACKTEST_START_DATE, BACKTEST_END_DATE, INITIAL_BALANCE
from datetime import datetime

def run_backtest_analysis():
    # Run backtest
    start_date = datetime.strptime(BACKTEST_START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(BACKTEST_END_DATE, "%Y-%m-%d")
    result = backtest(start_date, end_date, INITIAL_BALANCE)

    # Create DataFrame from trades
    trades_df = pd.DataFrame(result['trades'])
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df.set_index('date', inplace=True)

    # Calculate cumulative returns
    trades_df['cumulative_returns'] = (1 + trades_df['balance_change'].cumsum() / INITIAL_BALANCE)

    # Calculate drawdown
    trades_df['drawdown'] = (trades_df['cumulative_returns'] / trades_df['cumulative_returns'].cummax() - 1)

    # Calculate various metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['balance_change'] > 0])
    losing_trades = len(trades_df[trades_df['balance_change'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    average_profit = trades_df['balance_change'].mean()
    profit_factor = abs(trades_df[trades_df['balance_change'] > 0]['balance_change'].sum() / 
                        trades_df[trades_df['balance_change'] < 0]['balance_change'].sum()) if losing_trades > 0 else np.inf
    max_drawdown = trades_df['drawdown'].min()
    sharpe_ratio = (trades_df['balance_change'].mean() / trades_df['balance_change'].std()) * np.sqrt(252)  # Assuming daily trading

    # Print metrics
    print(f"Backtest Results ({start_date.date()} to {end_date.date()}):")
    print(f"Initial Balance: {INITIAL_BALANCE:,.0f} KRW")
    print(f"Final Balance: {result['final_balance']:,.0f} KRW")
    print(f"Total Return: {(result['final_balance'] - INITIAL_BALANCE) / INITIAL_BALANCE * 100:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Profit per Trade: {average_profit:,.0f} KRW")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(trades_df.index, trades_df['cumulative_returns'])
    plt.title('Cumulative Returns')
    plt.ylabel('Returns')

    plt.subplot(2, 1, 2)
    plt.fill_between(trades_df.index, trades_df['drawdown'], 0, alpha=0.3)
    plt.title('Drawdown')
    plt.ylabel('Drawdown')

    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.close()

    print(f"Backtest analysis chart saved as 'backtest_results.png'")

if __name__ == "__main__":
    run_backtest_analysis()