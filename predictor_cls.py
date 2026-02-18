import joblib
from Data_fetch import fetch_stock_bars
from feature_engineer import engineer_features


def predict_direction(symbol="AAPL", days_back=120, model_file="cls_aapl_model.joblib"):
    model_data = joblib.load(model_file)

    df_raw = fetch_stock_bars(symbol, days_back=days_back)
    df_feat = engineer_features(df_raw)

    feature_cols = model_data["feature_cols"]
    latest = df_feat[feature_cols].iloc[-1:].values

    scaler = model_data["scaler"]
    X_latest = scaler.transform(latest)

    model = model_data["models"]["rf_cls"]
    proba_up = model.predict_proba(X_latest)[0, 1]
    pred_label = int(proba_up >= 0.5)

    current_price = df_raw["close"].iloc[-1]

    print(f"Current {symbol} close: ${current_price:.2f}")
    print(f"Predicted direction (next day): {'UP' if pred_label == 1 else 'DOWN/FLAT'}")
    print(f"Probability up: {proba_up:.3f}")

    return {
        "current_price": current_price,
        "direction": pred_label,
        "proba_up": proba_up,
    }


if __name__ == "__main__":
    predict_direction()

