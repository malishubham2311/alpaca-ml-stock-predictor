import joblib
import pandas as pd
from datetime import datetime, timezone, timedelta

from Data_fetch import fetch_stock_bars
from feature_engineer import engineer_features


def save_model(model_data, filename="rf_aapl_model.joblib"):
    """Save trained model + scaler + features"""
    joblib.dump(model_data, filename)
    print(f"✅ Model saved: {filename}")


def load_model(filename="rf_aapl_model.joblib"):
    """Load trained model"""
    return joblib.load(filename)


def predict_next_close(symbol="AAPL", days_back=120, model_filename="rf_aapl_model.joblib"):
    """
    Fetch latest data → engineer features → predict next close price
    """
    print(f"🔄 Predicting next {symbol} close...")
    
    # Get latest data
    df_raw = fetch_stock_bars(symbol, days_back=days_back)
    print(f"📊 Latest close: ${df_raw['close'].iloc[-1]:.2f}")
    
    # Engineer features (latest row only)
    df_features = engineer_features(df_raw)
    latest_features = df_features.drop(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "target"]).iloc[-1:]
    
    # Load model
    model_data = load_model(model_filename)
    scaler = model_data["scaler"]
    model = model_data["models"]["rf"]
    feature_cols = model_data["feature_cols"]
    
    # Scale + predict
    latest_scaled = scaler.transform(latest_features.values.reshape(1, -1))[0]
    pred = model.predict([latest_scaled])[0]
    
    change = pred - df_raw['close'].iloc[-1]
    pct_change = (change / df_raw['close'].iloc[-1]) * 100
    
    print(f"🎯 Predicted next close: ${pred:.2f}")
    print(f"📈 Change: ${change:+.2f} ({pct_change:+.1f}%)")
    
    return {
        "current": df_raw['close'].iloc[-1],
        "predicted": pred,
        "change": change,
        "pct_change": pct_change
    }


if __name__ == "__main__":
    # First save your trained model
    print("1. First run model_trainer.py to train, THEN run this.")
    # save_model(out)  # Uncomment after training
    
    # Make prediction
    result = predict_next_close()
