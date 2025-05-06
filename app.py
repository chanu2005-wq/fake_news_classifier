import streamlit as st
import pickle
import joblib
import numpy as np
import os

# Check if the model and vectorizer exist
model_path = "model/model.pkl"
vectorizer_path = "model/vectorizer.pkl"

# Load model using pickle (ensure the path is correct)
try:
    with open(model_path, "rb") as f:
        model_pickle = pickle.load(f)
    st.write("Model loaded using pickle.")
except Exception as e:
    st.write(f"Error loading pickle model: {e}")

# Load model using joblib (ensure the path is correct)
try:
    model_joblib = joblib.load("model/model_joblib.pkl")
    st.write("Model loaded using joblib.")
except Exception as e:
    st.write(f"Error loading joblib model: {e}")

# Load the saved vectorizer (ensure the path is correct)
try:
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    st.write(f"Error loading vectorizer: {e}")

# Function to preprocess text using the loaded vectorizer
def preprocess_text(text):
    return vectorizer.transform([text])  # Transform text to the format the model expects

# Function to predict using the loaded model
def make_prediction(features):
    preprocessed_input = preprocess_text(features)
    prediction = model_pickle.predict(preprocessed_input)  # You can replace model_pickle with model_joblib
    return prediction[0]

# Streamlit UI to take user input
st.title("Fake News Classifier")

# File uploader for news article (txt, pdf, docx)
uploaded_file = st.file_uploader("Upload a News Article", type=["txt", "pdf", "docx"])

# Handle uploaded file
if uploaded_file is not None:
    # Read the file content
    file_content = uploaded_file.read().decode("utf-8")
    st.text_area("News Article", file_content, height=200)
    user_input = file_content  # Use the file content for prediction
else:
    # If no file uploaded, take text input directly
    user_input = st.text_area("Enter News Text", "", height=200)

# Make prediction when the button is pressed
if st.button('Predict'):
    if user_input:
        prediction = make_prediction(user_input)
        if prediction == 1:
            st.write("The news article is likely **FAKE**.")
        else:
            st.write("The news article is likely **REAL**.")
    else:
        st.write("Please enter text or upload a news article.")

# Optional: Add feedback form
feedback = st.text_area("Please provide feedback on the prediction:")
if st.button('Submit Feedback'):
    if feedback:
        # Store feedback in a text file (or send it to a database)
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")
        st.write("Thank you for your feedback!")
    else:
        st.write("Feedback cannot be empty!")


