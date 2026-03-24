import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = load_model("best_lstm_model.keras")

# ✅ BEST PRACTICE: Load trained scaler
scaler = joblib.load("scaler.save")

st.title("Tesla Stock Price Prediction")

# -------------------------------
# Load last sequence (IMPORTANT)
# -------------------------------
# BEST: load from saved file if available
# Example:
# last_sequence = joblib.load("last_sequence.save")

# TEMP (if not saved yet)
last_sequence = np.random.rand(60)

# -------------------------------
# Prediction Function
# -------------------------------
def multi_step_predict(model, last_sequence, steps):
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, -1, 1))[0][0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    return predictions

# -------------------------------
# UI
# -------------------------------
st.subheader("Predict Future Tesla Stock Prices")

days = st.slider("Select number of days to predict", 1, 10)

if st.button("Predict"):

    preds = multi_step_predict(model, last_sequence, days)

    # -------------------------------
    # Convert to real stock prices
    # -------------------------------
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("Predicted Prices:")

    for i, p in enumerate(preds):
        st.write(f"Day {i+1}: ${p[0]:.2f}")

    # -------------------------------
    # Plot Graph
    # -------------------------------
    fig, ax = plt.subplots()
    ax.plot(preds, marker='o')
    ax.set_title("Future Tesla Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

    st.pyplot(fig)