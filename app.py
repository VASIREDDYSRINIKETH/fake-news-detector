from flask import Flask, render_template, request
import joblib

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['news']
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        result = "REAL" if prediction == 1 else "FAKE"
        return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
