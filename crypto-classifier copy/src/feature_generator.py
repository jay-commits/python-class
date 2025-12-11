# src/feature_engineering.py
import pandas as pd
import ta
import os

def create_features():
    df = pd.read_csv("../data/raw/btcusdt.csv", parse_dates=["open_time"])
    
    # Technical indicators
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["stoch_k"] = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
    df["daily_return"] = df["close"].pct_change()
    df["volatility_30d"] = df["daily_return"].rolling(30).std()
    
    # Use only past data
    df["prev_close"] = df["close"].shift(1)
    df["prev_volume"] = df["volume"].shift(1)
    
    features = ['prev_close','prev_volume','rsi','macd','sma_20','sma_50','sma_200',
                'bb_upper','bb_lower','stoch_k','daily_return','volatility_30d']
    
    df[features] = df[features].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv("../data/processed/btcusdt_final.csv", index=False)
    print(f"Features created: {len(df)} rows")

if __name__ == "__main__":
    create_features()