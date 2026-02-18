from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import Config

def get_data_client():
    Config.validate()
    return StockHistoricalDataClient(
        api_key=Config.ALPACA_API_KEY,
        secret_key=Config.ALPACA_SECRET_KEY,
    )

def fetch_stock_bars(symbol, days_back=1095, timeframe=TimeFrame.Day, feed="iex"):
    """
    Fetch recent stock data using IEX (works with Basic plan).
    """
    client = get_data_client()
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=feed  # IEX = free data
    )
    
    bars = client.get_stock_bars(request_params)
    df = bars.df.reset_index()
    return df

if __name__ == "__main__":
    print("Fetching AAPL data (last 60 days, IEX feed)...")
    df = fetch_stock_bars("AAPL", days_back=60)
    
    print("SUCCESS! Data shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nLast date:", df['timestamp'].max())
