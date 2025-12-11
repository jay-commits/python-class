# src/predict.py
import joblib
import pandas as pd

def predict():
    model = joblib.load("../models/catboost.pkl")
    sample = pd.DataFrame([{
        'prev_close': 68000, 'prev_volume': 35000, 'rsi': 58, 'macd': 500,
        'sma_20': 67000, 'sma_50': 64500, 'sma_200': 60000,
        'bb_upper': 69500, 'bb_lower': 63500, 'stoch_k': 72,
        'daily_return': 0.02, 'volatility_30d': 0.04
    }])
    
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]
    labels = ["SELL", "HOLD", "BUY"]
    print(f"PREDICTION: {labels[int(pred)]}")
    print(f"Confidence → Buy: {prob[2]:.1%} | Hold: {prob[1]:.1%} | Sell: {prob[0]:.1%}")

if __name__ == "__main__":
    predict()