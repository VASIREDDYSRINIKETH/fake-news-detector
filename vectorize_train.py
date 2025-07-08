import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load cleaned data from step 1
train_df = pd.read_csv("train.tsv", sep="\t", header=None)
train_df.columns = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state_info", "party", "barely_true",
    "false", "half_true", "mostly_true", "pants_on_fire", "context"
]
df = train_df[["statement", "label"]].dropna()

def map_label(label):
    return 0 if label in ['false', 'pants-fire'] else 1

df["label"] = df["label"].apply(map_label)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['statement'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved!")
