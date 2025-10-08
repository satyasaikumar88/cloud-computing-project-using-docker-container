import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Drop missing values
    df = df.dropna()
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['polarity'].value_counts(normalize=True))
    
    return df

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and fit TF-IDF vectorizer
    print("\nFitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        stop_words='english'
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Initialize and train the model
    print("Training model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate on test set
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    return model, tfidf

def save_artifacts(model, vectorizer, model_path='model.joblib', vectorizer_path='tfidf_vectorizer.joblib'):
    """Save the trained model and vectorizer"""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def main():
    # File path - update this to your dataset location
    data_path = 'aspect_sentiment_large.csv'
    
    print(f"Loading data from {data_path}...")
    df = load_and_preprocess_data(data_path)
    
    # Combine sentence and aspect for better context
    X = df['sentence'] + ' ' + df['aspect_term']
    y = df['polarity']
    
    # Train the model
    model, vectorizer = train_model(X, y)
    
    # Save the model and vectorizer
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()
