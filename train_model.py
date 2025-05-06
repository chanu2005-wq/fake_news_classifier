import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Define file paths
fake_file_path = "data/Fake.csv"
real_file_path = "data/True.csv"

# Check if files exist
if os.path.exists(fake_file_path) and os.path.exists(real_file_path):
    # Load datasets
    fake_df = pd.read_csv(fake_file_path)
    real_df = pd.read_csv(real_file_path)

    # Assign labels
    fake_df['label'] = 0  # Fake
    real_df['label'] = 1  # Real

    # Combine and shuffle
    df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=42)

    # Features and labels
    X = df['text']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Create model directory if not exists
    os.makedirs('model', exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

    print("Model and vectorizer saved successfully!")

else:
    print("❌ Error: One or both dataset files not found. Check paths: 'data/Fake.csv', 'data/True.csv'")

    







