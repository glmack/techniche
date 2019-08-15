import pickle
from gensim.corpora import Dictionary
from flask import Flask, request, render_template, jsonify
from topic_model import get_topics

with open('model_4.pkl', 'rb') as f_1:
    model = pickle.load(f_1)

with open('id_to_word_1000train.pkl', 'rb') as f_2:
    id_to_word = pickle.load(f_2)

app = Flask(__name__, static_url_path="")

@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a prediction of topics"""
    data = request.json
    # prediction = model.predict_proba([data['user_input']]) #ME example
    # print(data)
    prediction = get_topics(id_to_word.doc2bow([data['user_input']]), model, k=15)
    # round_prediction = round(prediction[0][1], 2)
    return jsonify({'prediction': prediction})
    # return jsonify(data["user_input"]