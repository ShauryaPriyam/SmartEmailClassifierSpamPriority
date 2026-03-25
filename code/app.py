from flask import Flask, request, render_template
import joblib
import json
import numpy as np

# NLP
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# SHAP
import shap

app = Flask(__name__)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    
    words = word_tokenize(text)
    
    processed = []
    for word in words:
        if word.isalpha() and word not in stop_words:
            stemmed = ps.stem(word)
            processed.append(stemmed)
    
    return processed


# 🔹 Load models
models = {
    "Logistic Regression": joblib.load("../models/model.pkl"),
    "SVM": joblib.load("../models/svm.pkl"),
    "Naive Bayes": joblib.load("../models/Naive_bayes.pkl"),
    "Random Forest": joblib.load("../models/random_forest.pkl"),
    "KNN": joblib.load("../models/knn.pkl")
}

# 🔹 SHAP function
def get_shap(model, text):
    try:
        vectorizer = model.named_steps['tfidf']
        clf = model.named_steps['clf']

        X = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        explainer = shap.LinearExplainer(clf, X)
        shap_values = explainer.shap_values(X)

        vals = shap_values[0]
        idx = np.argsort(np.abs(vals))[-10:]

        result = []
        for i in idx:
            result.append({
                "word": feature_names[i],
                "value": float(vals[i])
            })
        return result
    except:
        return []


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None
    text = ""
    selected_model = "Logistic Regression"

    all_results = {}
    shap_data = []

    if request.method == 'POST':
        text = request.form['email']
        selected_model = request.form['model']

        # 🔹 Selected model
        model = models[selected_model]

        pred = model.predict([text])[0]

        prediction = "spam" if pred == 1 else "not-spam"

        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba([text])[0]))

        # 🔥 MODEL COMPARISON
        for name, m in models.items():
            p = m.predict([text])[0]
            c = None
            if hasattr(m, "predict_proba"):
                c = float(max(m.predict_proba([text])[0]))

            all_results[name] = {
                "prediction": "spam" if p == 1 else "not-spam",
                "confidence": round(c * 100, 1) if c else None
            }

        # 🔥 SHAP
        shap_data = get_shap(model, text)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        model=selected_model,
        models=models.keys(),
        text=text,
        all_results=json.dumps(all_results),
        shap_data=json.dumps(shap_data)
    )


if __name__ == "__main__":
    app.run(debug=True)