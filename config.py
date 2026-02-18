import os
from dotenv import load_dotenv

#load env variables from .env
load_dotenv()

class Config:
    # Alpaca API credentials
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = os.getenv(
        "ALPACA_BASE_URL",
        "https://paper-api.alpaca.markets"
    )

    #Model parameters
    Test_size = 0.2
    Ramdon_state = 42

    #Data parameters
    Default_symbol = "AAPL"
    Lookback_days = 730 #2 years

    #feature engineering parameters
    SMA_windows = [5, 10, 20, 50]
    EMA_windows = [5, 10, 20]
    volatility_window = [5, 20]
    Lag_periods = [1, 2, 3, 5, 10]

    #Directory 
    Model_Dir = "Models"
    Data_Dir = "Data"

    @classmethod
    def validate(cls):
        #check if all required keys are present
        if not cls.ALPACA_API_KEY or cls.ALPACA_API_KEY == "your_api_key_here":
            raise ValueError("Please set ALPACA_API_KEY in your .env file")
        if not cls.ALPACA_SECRET_KEY or cls.ALPACA_SECRET_KEY == "your_secret_key_here":
            raise ValueError("Please set ALPACA_SECRET_KEY in your .env file")
        return True