import streamlit as st
import numpy as np
from keras.models import load_model

# Load model
model = load_model("best_lstm_model.keras")

st.title("Tesla Stock Price Prediction")

# Dummy last sequence (replace with real one if available)
last_sequence = np.random.rand(60)

# ✅ Function (YOU MISSED THIS)
def multi_step_predict(model, last_sequence, steps):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, -1, 1))[0][0]
        predictions.append(pred)

        current_seq = np.append(current_seq[1:], pred)

    return predictions

# UI
days = st.slider("Select number of days to predict", 1, 10)

if st.button("Predict"):
    preds = multi_step_predict(model, last_sequence, days)
    st.write(preds)