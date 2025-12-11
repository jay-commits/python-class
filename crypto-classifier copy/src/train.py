# src/train.py
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

FEATURES = ['prev_close','prev_volume','rsi','macd','sma_20','sma_50','sma_200',
            'bb_upper','bb_lower','stoch_k','daily_return','volatility_30d']

def train_models():
    df = pd.read_csv("../data/processed/btcusdt_final.csv")
    df["future_return"] = df["close"].pct_change().shift(-1)
    df["label"] = df["future_return"].apply(lambda x: 2 if x > 0.02 else (0 if x < -0.02 else 1))
    df = df.dropna().reset_index(drop=True)
    
    X = df[FEATURES]
    y = df["label"]
    
    # Class weights — fixed!
    weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y)
    weight_list = weights.tolist()
    print("Class weights [Sell, Hold, Buy]:", [f"{w:.2f}" for w in weight_list])
    
    split = int(len(df) * 0.85)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    
    models = {
        "catboost": CatBoostClassifier(iterations=1500, depth=6, class_weights=weight_list, verbose=0),
        "lightgbm": LGBMClassifier(class_weight='balanced'),
        "xgboost": XGBClassifier(scale_pos_weight=weight_list[2], eval_metric="mlogloss"),
        "random_forest": RandomForestClassifier(class_weight='balanced', n_estimators=800),
        "logistic": LogisticRegression(class_weight='balanced', max_iter=1000)
    }
    
    os.makedirs("../models", exist_ok=True)
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"../models/{name}.pkl")
        print(f"→ {name}.pkl saved!")
    
    print("\nTRAINING COMPLETE — ALL MODELS READY!")

if __name__ == "__main__":
    train_models()