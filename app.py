# streamlit drives the UI
import streamlit as st
# import numpy for numeric arrays
import numpy as np
# import matplotlib for plotting results
import matplotlib.pyplot as plt
# load_model retrieves the saved LSTM network
from keras.models import load_model
# joblib reuses saved scalers and sequences
import joblib

# Streamlit UI that loads the trained LSTM and scaler to serve Tesla forecasts.

# -------------------------------
# Page Title
# -------------------------------
# Give the page a quick prompt about what the app does.
# set the Streamlit title for the sidebar
st.title("Tesla Stock Price Prediction")

# -------------------------------
# Load Model & Scaler
# -------------------------------
# load the trained LSTM from disk
model = load_model("best_lstm_model.keras")

# ✅ BEST PRACTICE: Load trained scaler
# bring back the scaler used during training
scaler = joblib.load("scaler.save")

# -------------------------------
# Load last sequence (IMPORTANT)
# -------------------------------
# BEST: load from saved file if available
# Example:
# last_sequence = joblib.load("last_sequence.save")

# TEMP (if not saved yet)
# fill the buffer with random numbers while waiting for saved data
last_sequence = np.random.rand(60)

# -------------------------------
# Prediction Function
# -------------------------------
# Helper that feeds each prediction back into the input sequence for multi-step forecasts.
def multi_step_predict(model, last_sequence, steps):
    # collect each prediction to return later
    predictions = []
    # work on a copy so we never mutate the stored window
    current_seq = last_sequence.copy()

    # iteratively forecast 'steps' values
    for _ in range(steps):
        # predict the next value from the latest window
        pred = model.predict(current_seq.reshape(1, -1, 1))[0][0]
        # add the forecast to the list
        predictions.append(pred)
        # slide the window and include the new prediction
        current_seq = np.append(current_seq[1:], pred)

    # hand the generated sequence of predictions back
    return predictions

# -------------------------------
# UI
# -------------------------------
# describe the purpose of this section to visitors
# render the subheader introducing the controls
st.subheader("Predict Future Tesla Stock Prices")

# Let visitors dial in how many future days they want to forecast.
days = st.slider("Select number of days to predict", 1, 10)

# run forecasting logic when the button is pressed
if st.button("Predict"):

    # Trigger predictions whenever the button is pressed.
    preds = multi_step_predict(model, last_sequence, days)

    # -------------------------------
    # Convert to real stock prices
    # -------------------------------
    # scale every forecast back into price space
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # -------------------------------
    # Display Results
    # -------------------------------
    st.subheader("Predicted Prices:")

    # show each day with its estimated price
    for i, p in enumerate(preds):
        st.write(f"Day {i+1}: ${p[0]:.2f}")

    # -------------------------------
    # Plot Graph
    # -------------------------------
    # prepare the figure for plotting
    fig, ax = plt.subplots()
    # draw the predicted trajectory
    ax.plot(preds, marker='o')
    # label the chart for clarity
    ax.set_title("Future Tesla Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

    # hand the figure off to Streamlit
    st.pyplot(fig)
