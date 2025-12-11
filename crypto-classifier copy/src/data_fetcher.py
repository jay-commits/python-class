# src/data_fetcher.py
import requests
import pandas as pd
import os

def fetch_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 2000}
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    
    os.makedirs("../data/raw", exist_ok=True)
    df.to_csv("../data/raw/btcusdt.csv", index=False)
    print("Data fetched and saved!")

if __name__ == "__main__":
    fetch_data()