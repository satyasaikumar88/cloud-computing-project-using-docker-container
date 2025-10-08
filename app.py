from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
print("Loading model and vectorizer...")
try:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model/vectorizer: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'sentence' not in data or 'aspect' not in data:
            return jsonify({
                'error': 'Invalid input. Please provide both "sentence" and "aspect" in the request body.'
            }), 400
        
        # Combine sentence and aspect for prediction
        text = f"{data['sentence']} {data['aspect']}"
        
        # Transform input using the loaded vectorizer
        X = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return the result
        return jsonify({
            'sentence': data['sentence'],
            'aspect': data['aspect'],
            'predicted_polarity': prediction
        })
        
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
