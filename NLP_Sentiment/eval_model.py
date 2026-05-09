import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

csv_path = r"C:\Users\narla\Downloads\archive (3)\Mobile Reviews Sentiment.csv"
df = pd.read_csv(csv_path)
df = df[['review_text', 'sentiment']].dropna()
df = df.drop_duplicates(subset="review_text")
df["clean_text"] = df["review_text"].apply(preprocess)
df = df.drop_duplicates(subset="clean_text")

# Load model and vectorizer
model = joblib.load("trained_model.pkl")
vectorizer = joblib.load("trained_vectorizer.pkl")

X = vectorizer.transform(df["clean_text"])
y = df["sentiment"]

y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# Test specific phrases
test_phrases = ["the battery life is too good", "terrible phone, it broke", "amazing camera", "battery life good"]
clean_test = [preprocess(t) for t in test_phrases]
X_test = vectorizer.transform(clean_test)
preds = model.predict(X_test)

for p, c, pred in zip(test_phrases, clean_test, preds):
    print(f"Phrase: '{p}' | Clean: '{c}' | Pred: {pred}")

