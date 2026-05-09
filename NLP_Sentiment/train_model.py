import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load data
csv_path = r"C:\Users\narla\Downloads\archive (3)\Mobile Reviews Sentiment.csv"
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# Keep only needed columns
df = df[['review_text', 'sentiment']]
df = df.drop_duplicates(subset="review_text")

print("Preprocessing text...")
df["clean_text"] = df["review_text"].apply(preprocess)
df = df.drop_duplicates(subset="clean_text")

# --- Augment Dataset with Basic Vocabulary ---
basic_data = pd.DataFrame({
    'review_text': [
        "the battery life is too good", "excellent amazing great perfect good", "love it best phone",
        "terrible bad worst awful broken", "glitchy hate it poor quality", "battery is bad",
        "it is okay fine average", "not good not bad"
    ],
    'sentiment': [
        "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative",
        "Neutral", "Neutral"
    ]
})
basic_data["clean_text"] = basic_data["review_text"].apply(preprocess)
df = pd.concat([df, basic_data], ignore_index=True)
# -------------------------------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.4,
    random_state=42,
    stratify=df["sentiment"]
)

print("Vectorizing...")
tfidf = TfidfVectorizer(
    max_features=2500,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)
X_train = tfidf.fit_transform(X_train_text)

from sklearn.linear_model import LogisticRegression

print("Training model...")
model = LogisticRegression(random_state=42, C=1.0)
model.fit(X_train, y_train)

# Save
print("Saving model and vectorizer...")
joblib.dump(model, "trained_model.pkl")
joblib.dump(tfidf, "trained_vectorizer.pkl")

print("Done!")
