import pickle
from gensim.corpora import Dictionary
from flask import Flask, request, render_template, jsonify
from topic_model import get_topics

with open('model_4.pkl', 'rb') as f:
    model = pickle.load(f)

with open('id_to_word_1000train.pkl', 'rb') as f:
    id_to_word = pickle.load(f)

app = Flask(__name__, static_url_path="")


@app.route("/")
def index():
    """Return the main page"""
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return prediction of important topics in new text string"""
    data = request.json
    prediction = get_topics(id_to_word.doc2bow([data['user_input']]),
                            model,
                            k=3)
    return jsonify({'prediction': prediction})