import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Constants ---
MAX_LEN = 200

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # streamlit_app/
CHECKPOINT_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'checkpoints'))

# --- MODEL MAP ---
MODEL_MAP = {
    "Vanilla RNN": (
        os.path.join(CHECKPOINT_DIR, "rnn_model.h5"),
        os.path.join(CHECKPOINT_DIR, "rnn_tokenizer.pkl"),
        "A basic RNN model using SimpleRNN layers."
    ),
    "LSTM": (
        os.path.join(CHECKPOINT_DIR, "lstm_model.h5"),
        os.path.join(CHECKPOINT_DIR, "lstm_tokenizer.pkl"),
        "An LSTM model that handles long-term dependencies."
    ),
    "BiLSTM": (
        os.path.join(CHECKPOINT_DIR, "bilstm_model.h5"),
        os.path.join(CHECKPOINT_DIR, "bilstm_tokenizer.pkl"),
        "A Bidirectional LSTM for richer context."
    ),
    "GRU": (
        os.path.join(CHECKPOINT_DIR, "gru_model.h5"),
        os.path.join(CHECKPOINT_DIR, "gru_tokenizer.pkl"),
        "A faster alternative to LSTM using GRU units."
    ),
    "BiGRU": (
        os.path.join(CHECKPOINT_DIR, "bigru_model.h5"),
        os.path.join(CHECKPOINT_DIR, "bigru_tokenizer.pkl"),
        "A Bidirectional GRU model."
    )
}

# --- Page Setup ---
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎭 Sentiment Analysis of Movie Reviews")
st.write("Select a model, enter a movie review, and view the predicted sentiment.")

# --- Sidebar ---
model_choice = st.sidebar.selectbox("Choose Model", list(MODEL_MAP.keys()))
model_path, tokenizer_path, model_description = MODEL_MAP[model_choice]

st.sidebar.markdown(f"ℹ️ **Model Info:** {model_description}")

# --- Input Box ---
review = st.text_area("Enter IMDb movie review text:", height=150)

# --- Prediction Logic ---
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        # Load model and tokenizer
        with st.spinner("🔄 Loading model..."):
            try:
                model = tf.keras.models.load_model(model_path)
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            except Exception as e:
                st.error(f"❌ Error loading model or tokenizer: {e}")
                st.stop()

        # Preprocess
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        pred = model.predict(padded)[0][0]
        label = "🟢 Positive" if pred >= 0.5 else "🔴 Negative"
        confidence = float(pred) if pred >= 0.5 else 1 - float(pred)

        # Display results
        st.markdown(f"### 📢 Sentiment: {label}")
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
        st.success("✅ Prediction complete.")