import streamlit as st
import pandas as pd
from datetime import datetime

def load_trade_history():
    # This function should load your trade history data
    # For now, we'll use a dummy data generator
    dummy_data = [
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Buy", 2.5, "Strong buy signal", 1.5, 1000000, 2000000, 2100000),
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Sell", -1.2, "Weak market", 1.3, 1050000, 1950000, 2050000),
        # Add more dummy data as needed
    ]
    return dummy_data

def main():
    st.title("Ethereum Trading Bot Dashboard")

    # Load trade history
    trades = load_trade_history()

    # Convert trades to a DataFrame
    df = pd.DataFrame(trades, columns=[
        "Timestamp", "Decision", "Percentage", "Reason", 
        "ETH Balance", "KRW Balance", "ETH Average Buy Price", "ETH/KRW Price"
    ])

    # Display the trade history as a table
    st.header("Trade History")
    st.dataframe(df)

    # Add some basic statistics
    st.header("Trading Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", len(df))
    with col2:
        st.metric("Buy Trades", len(df[df['Decision'] == 'Buy']))
    with col3:
        st.metric("Sell Trades", len(df[df['Decision'] == 'Sell']))

    # Add a line chart for ETH/KRW price over time
    st.header("ETH/KRW Price Trend")
    st.line_chart(df.set_index('Timestamp')['ETH/KRW Price'])

    # Add a bar chart for ETH and KRW balances
    st.header("Account Balance")
    balance_df = df[['Timestamp', 'ETH Balance', 'KRW Balance']].set_index('Timestamp')
    st.bar_chart(balance_df)

if __name__ == "__main__":
    main()