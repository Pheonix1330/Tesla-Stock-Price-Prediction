import streamlit as st
import numpy as np
from keras.models import load_model

model = load_model("best_lstm_model.keras")

st.title("Tesla Stock Price Prediction")

days = st.slider("Select number of days to predict", 1, 10)

if st.button("Predict"):
    preds = multi_step_predict(model, last_sequence, days)
    st.write(preds)