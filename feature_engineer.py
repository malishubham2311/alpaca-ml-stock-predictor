import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Take raw Alpaca bars df (with columns like timestamp, open, high, low, close, volume)
    and return a new df with technical features + target (next close).
    """

    df = df.copy()
    df = df.sort_values("timestamp")

    #Basic returns
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    #Simple Moving Averages
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()

    #price vs MA
    df["price_vs_sma_5"] = df["close"] / df["sma_5"] - 1
    df["price_vs_sma_20"] = df["close"] / df["sma_20"] - 1

    #Volatility
    df["volatility_5"] = df["return"].rolling(5).std()
    df["volatility_20"] = df["return"].rolling(20).std()

    #Momentum
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_20"] = df["close"] - df["close"].shift(20)

    #RSI (14 days)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    #Volume features
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] =  df["volume"] / df["vol_sma_20"]

    #lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    #Target: next day's close price
    #classification target (1 = up, 0 = down or flat)
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target_cls"] = (df["next_return"] > 0).astype(int)

    #Drop rows with NaN values (due to rolling calculations)
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    from Data_fetch import fetch_stock_bars  # or your exact filename/function

    df_raw = fetch_stock_bars("AAPL", days_back=120)
    print("Raw shape:", df_raw.shape)

    df_feat = engineer_features(df_raw)
    print("With features shape:", df_feat.shape)
    print(df_feat.head())
