import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import pandas as pd

from Data_fetch import fetch_stock_bars
from feature_engineer import engineer_features
import joblib

# Page config
st.set_page_config(
    page_title="Stock Predictor ML",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 ML Stock Price Predictor")
st.markdown("**Production ML pipeline**: Alpaca → Features → RandomForest → Predict")

@st.cache_data
def load_model():
    return joblib.load("rf_aapl_model.joblib")

def predict_symbol(symbol, days_back=120):
    model_data = load_model()
    
    # Fetch data
    df_raw = fetch_stock_bars(symbol, days_back=days_back)
    
    # Features
    df_features = engineer_features(df_raw)
    
    # Latest features
    feature_cols = model_data["feature_cols"]
    latest_features = df_features[feature_cols].iloc[-1:].values
    
    # Predict
    scaler = model_data["scaler"]
    model = model_data["models"]["rf"]
    
    latest_scaled = scaler.transform(latest_features)
    pred = model.predict(latest_scaled)[0]
    
    return {
        "current": df_raw["close"].iloc[-1],
        "predicted": pred,
        "df_raw": df_raw,
        "df_features": df_features
    }

# Sidebar
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
days_back = st.sidebar.slider("Days of history", 30, 365, 120)

if st.sidebar.button("🔮 Predict!", type="primary"):
    with st.spinner("Computing prediction..."):
        result = predict_symbol(symbol, days_back)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Current Close",
                f"${result['current']:.2f}"
            )
            st.metric(
                "Predicted Next Close",
                f"${result['predicted']:.2f}",
                delta=f"${result['predicted'] - result['current']:+.2f}"
            )
        
        with col2:
            change_pct = ((result['predicted'] - result['current']) / result['current']) * 100
            st.metric("Change", f"{change_pct:+.1f}%")
        
        # Plot historical data
        fig = px.line(
            result['df_raw'], 
            x="timestamp", 
            y="close", 
            title=f"{symbol} Historical Close Prices"
        )
        fig.add_hline(
            y=result['predicted'], 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Predicted: ${result['predicted']:.2f}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Prediction complete!")

# Model info
with st.expander("📊 Model Performance (Test Set)"):
    model_data = load_model()
    results_df = pd.DataFrame(model_data["results"]).T
    st.dataframe(results_df.style.format({"rmse": "${:.2f}", "mae": "${:.2f}"}))
