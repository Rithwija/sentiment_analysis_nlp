import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

!pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("Mobile Reviews Sentiment.csv")

# Keep only needed columns
df = df[['review_text', 'sentiment']]

# Remove duplicates
df = df.drop_duplicates(subset="review_text")

print(df.shape)
print(df['sentiment'].value_counts())

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["review_text"].apply(preprocess)

df = df.drop_duplicates(subset="clean_text")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"],
    df["sentiment"],
    test_size=0.4,
    random_state=42,
    stratify=df["sentiment"]
)

tfidf = TfidfVectorizer(
    max_features=2500,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9

)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

df = df.sample(frac=0.8, random_state=42)

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
}


results = {}

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    results[name] = test_acc

    print(f"\n{name}")
    print("Training Accuracy :", round(train_acc, 4))
    print("Testing Accuracy  :", round(test_acc, 4))

final_model = model

from sklearn.metrics import confusion_matrix

y_pred = final_model.predict(X_test)   # ✅ ADD THIS LINE

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 🔹 LDA Topic Modeling (Improved Version)
# =========================

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Convert text into count-based features (better for LDA)
count_vectorizer = CountVectorizer(max_features=2500, stop_words='english')
X_counts = count_vectorizer.fit_transform(df["clean_text"])

# Step 2: Train LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_counts)

# Step 3: Display topics with top words
terms = count_vectorizer.get_feature_names_out()

print("\n🔍 Extracted Topics:\n")

for i, topic in enumerate(lda.components_):
    top_words = [terms[i] for i in topic.argsort()[-10:]]
    print(f"Topic {i+1}: {top_words}")
    print()

# Step 4: Get topic distribution for each review
topic_distribution = lda.transform(X_counts)

# Step 5: Assign dominant topic to each review
df["dominant_topic"] = np.argmax(topic_distribution, axis=1)

# Step 6: Visualize topic distribution
plt.figure()
sns.countplot(x=df["dominant_topic"])
plt.title("Topic Distribution Across Reviews")
plt.xlabel("Topic Number")
plt.ylabel("Number of Reviews")
plt.show()

for i, topic in enumerate(lda.components_):
    words = [terms[i] for i in topic.argsort()[-20:]]
    text = " ".join(words)

    wc = WordCloud(background_color='white')

    plt.figure(figsize=(6,5))
    plt.imshow(wc.generate(text))
    plt.title(f"Topic {i+1} Word Cloud")
    plt.axis("off")
    plt.show()

# =========================
# 🔹 K-Means Clustering (Improved Version)
# =========================

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Step 1: Use TF-IDF features (better for clustering)
X_all = tfidf.fit_transform(df["clean_text"])

# Step 2: Choose number of clusters
k = 5

# Step 3: Train K-Means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_all)

# Step 4: Assign cluster labels to each review
df["cluster"] = kmeans.labels_

# Step 5: Display cluster distribution
plt.figure()
sns.countplot(x=df["cluster"])
plt.title("Cluster Distribution Across Reviews")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Reviews")
plt.show()

# Step 6: Show top words in each cluster (VERY IMPORTANT)
terms = tfidf.get_feature_names_out()

print("\n🔍 Top words per cluster:\n")

for i in range(k):
    center = kmeans.cluster_centers_[i]
    top_words = [terms[i] for i in center.argsort()[-10:]]
    print(f"Cluster {i}: {top_words}")
    print()

# Step 7: Compare clusters with sentiment (insightful analysis)
plt.figure()
sns.countplot(x=df["cluster"], hue=df["sentiment"])
plt.title("Cluster vs Sentiment")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get sorted clusters (0,1,2,3,4)
clusters = sorted(df["cluster"].unique())

for cluster in clusters:

    print(f"\n========== Cluster {cluster} ==========\n")

    plt.figure(figsize=(15,5))

    # Positive
    pos_text = " ".join(df[(df["cluster"] == cluster) &
                              (df["sentiment"] == "Positive")]["clean_text"])

    if len(pos_text) > 0:
        wc = WordCloud(background_color='white', colormap='Greens')
        plt.subplot(1,3,1)
        plt.imshow(wc.generate(pos_text))
        plt.title("Positive")
        plt.axis("off")

    # Negative
    neg_text = " ".join(df[(df["cluster"] == cluster) &
                              (df["sentiment"] == "Negative")]["clean_text"])

    if len(neg_text) > 0:
        wc = WordCloud(background_color='white', colormap='Reds')
        plt.subplot(1,3,2)
        plt.imshow(wc.generate(neg_text))
        plt.title("Negative")
        plt.axis("off")

    # Neutral
    neu_text = " ".join(df[(df["cluster"] == cluster) &
                              (df["sentiment"] == "Neutral")]["clean_text"])

    if len(neu_text) > 0:
        wc = WordCloud(background_color='white', colormap='Blues')
        plt.subplot(1,3,3)
        plt.imshow(wc.generate(neu_text))
        plt.title("Neutral")
        plt.axis("off")

    plt.suptitle(f"Cluster {cluster} Word Clouds", fontsize=16)
    plt.show()

