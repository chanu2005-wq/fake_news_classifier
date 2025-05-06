import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Check the current working directory
print("Current Working Directory:", os.getcwd())

# Define the file paths for the fake and real news CSV files
fake_file_path = "data/Fake.csv"
real_file_path = "data/True.csv"

# Check if both files exist before trying to read them
if os.path.exists(fake_file_path) and os.path.exists(real_file_path):
    # Read both CSV files into DataFrames
    fake_df = pd.read_csv(fake_file_path)
    real_df = pd.read_csv(real_file_path)

    # Add a column to each DataFrame indicating the label (Fake/Real)
    fake_df['label'] = 0  # 0 for fake news
    real_df['label'] = 1  # 1 for real news

    # Concatenate the two DataFrames
    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Split data into features (X) and labels (y)
    X = df['text']  # Assuming the text column is named 'text'
    y = df['label']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data into numerical features using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the model (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Example: Train the model
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)  # X_train is your text data
    model = RandomForestClassifier()
    model.fit(X_train_tfidf, y_train)  # y_train is the target labels

    # Predict on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save the model and vectorizer
    joblib.dump(model, 'model/model_joblib.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

    # Save the trained model and vectorizer for future use
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
else:
    print("Error: One or both of the files were not found. Please check the file paths and try again.")
    import joblib






