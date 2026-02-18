import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from Data_fetch import fetch_stock_bars
from feature_engineer import engineer_features
from config import Config
import joblib


def prepare_data(symbol: str = None, days_back: int = 1095):  # Fixed default
    if symbol is None:
        symbol = "AAPL"

    print(f"Fetching {days_back} days of {symbol} data...")
    
    # 1) fetch raw bars
    df_raw = fetch_stock_bars(symbol, days_back=days_back)
    print(f"Raw data shape: {df_raw.shape}")

    # 2) engineer features
    df = engineer_features(df_raw)
    print(f"Features shape: {df.shape}")

    # 3) define X and y
    target_col = "target"
    drop_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume", target_col]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df[target_col].values

    # 4) time-ordered split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5) scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Fixed: use np.sqrt()
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE : ${mae:.2f}")
    print(f"R2  : {r2:.3f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_models():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = prepare_data()

    results = {}

    # 1) Linear regression
    lin = LinearRegression()
    lin.fit(X_train_scaled, y_train)
    results["LinearRegression"] = evaluate_model("LinearRegression", lin, X_test_scaled, y_test)

    # 2) Random forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,  # Hardcode for now
    )
    rf.fit(X_train_scaled, y_train)
    results["RandomForest"] = evaluate_model("RandomForest", rf, X_test_scaled, y_test)

    return {
        "results": results,
        "models": {"linear": lin, "rf": rf},
        "scaler": scaler,
        "feature_cols": feature_cols,
    }

def save_model(model_data, filename="rf_aapl_model.joblib"):
    """Save trained model + scaler + features"""
    joblib.dump(model_data, filename)
    print(f"✅ Model saved: {filename}")


if __name__ == "__main__":
    out = train_models()
    save_model(out)  # Add this import + function above
    print("\n🎉 Ready for predictions!")