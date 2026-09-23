import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Load Model and Tokenizer
# -----------------------------

model = load_model("spam_ham_bilstm_model.keras")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LENGTH = 189


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("📩 Spam / Ham SMS Classifier")

st.write(
    "Enter an SMS message below to check whether it is "
    "Spam or Ham."
)


# -----------------------------
# User Input
# -----------------------------

message = st.text_area(
    "Enter your SMS:",
    placeholder="Example: Congratulations! You have won a prize!"
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")

    else:
        # Convert text into sequence
        sequence = tokenizer.texts_to_sequences([message])

        # Apply padding
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_LENGTH,
            padding="post"
        )

        # Predict probability
        probability = model.predict(
            padded_sequence,
            verbose=0
        )[0][0]

        # Classification
        if probability >= 0.5:
            prediction = "Spam"
        else:
            prediction = "Ham"

        # Display result
        st.subheader("Prediction")

        if prediction == "Spam":
            st.error(f"🚨 {prediction}")
        else:
            st.success(f"✅ {prediction}")

        st.write(
            f"Spam Probability: **{probability:.2%}**"
        )