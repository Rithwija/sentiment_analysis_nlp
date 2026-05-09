from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

app = Flask(__name__)

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Preprocessing function from your notebook
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load the model and vectorizer
# Update these paths to match exactly where your files are
MODEL_PATH = r"trained_model.pkl"
VECTORIZER_PATH = r"trained_vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model, vectorizer = None, None

@app.route('/')
def home():
    return render_template('index.html')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded on server."}), 500

    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    review_text = data['text']
    
    # Preprocess
    clean_text = preprocess(review_text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([clean_text])
    
    # Predict with ML Model
    probabilities = model.predict_proba(vectorized_text)[0]
    max_prob = max(probabilities)
    ml_sentiment = model.classes_[probabilities.argmax()]
    
    # Hybrid Approach: If ML model is uncertain (e.g. unseen words/typos like "goodddd"), fallback to VADER
    if max_prob < 0.45 or vectorized_text.nnz == 0:
        vader_scores = sia.polarity_scores(review_text)
        compound = vader_scores['compound']
        # Adjusted thresholds to keep 'okay' neutral but 'good' positive
        if compound >= 0.43:
            sentiment = "Positive"
        elif compound <= -0.10:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
    else:
        sentiment = ml_sentiment
    
    return jsonify({
        "sentiment": sentiment,
        "clean_text": clean_text
    })

if __name__ == '__main__':
    app.run(debug=True)
