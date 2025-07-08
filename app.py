from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news = request.form["news"]
        vector = vectorizer.transform([news])
        result = model.predict(vector)[0]
        prediction = "Real News" if result == 1 else "Fake News"
    return render_template("index.html", prediction=prediction)