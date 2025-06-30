import streamlit as st
from sentiment_stock_predictor import run_prediction_pipeline

st.title("📈 Sentiment-Aware Stock Predictor")

ticker = st.text_input("Enter stock ticker:", value="AAPL")

if st.button("Run Model"):
    run_prediction_pipeline(ticker)
