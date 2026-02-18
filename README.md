# ML Stock Price Predictor (Alpaca + Streamlit)

End-to-end pipeline:
- Fetches OHLCV data from Alpaca Markets API
- Engineers technical features
- Trains:
  - Regression model (next-day close)
  - Classification model (next-day up / down)
- Exposes predictions via a Streamlit web UI

## Setup

```bash
git clone https://github.com/<your-username>/alpaca-ml-stock-predictor.git
cd alpaca-ml-stock-predictor
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate
pip install -r requirements.txt
