# Aspect-Based Sentiment Analysis (ABSA)

This project implements an Aspect-Based Sentiment Analysis system that predicts sentiment (positive, negative, or neutral) for specific aspects in restaurant reviews.

## Project Structure

```
absa_project/
├── app.py                 # Flask API for predictions
├── train_absa.py          # Model training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd absa_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your dataset named `aspect_sentiment_large.csv` in the project root
   - The CSV should have three columns: `sentence`, `aspect_term`, and `polarity`

## Training the Model

Run the training script:
```bash
python train_absa.py
```

This will:
1. Load and preprocess the data
2. Train a Logistic Regression model with TF-IDF features
3. Save the model and vectorizer to disk
4. Print evaluation metrics

## Running the API

Start the Flask API:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Make Predictions
**Endpoint:** `POST /predict`

**Request Body:**
```json
{
    "sentence": "The food was delicious but the service was slow.",
    "aspect": "food"
}
```

**Response:**
```json
{
    "sentence": "The food was delicious but the service was slow.",
    "aspect": "food",
    "predicted_polarity": "positive"
}
```

### 2. Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
    "status": "healthy"
}
```

## Testing with cURL

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"sentence":"The food was delicious but the service was slow.","aspect":"food"}'
```

## Future Enhancements

1. **Model Improvement**:
   - Fine-tune a BERT-based model for better accuracy
   - Experiment with different pre-trained language models
   - Implement cross-validation for more robust evaluation

2. **API Enhancements**:
   - Add authentication
   - Implement rate limiting
   - Add request validation
   - Add Swagger/OpenAPI documentation

3. **Deployment**:
   - Containerize with Docker
   - Set up CI/CD pipeline
   - Deploy to cloud platform (AWS/GCP/Azure)

## License

This project is licensed under the MIT License.
