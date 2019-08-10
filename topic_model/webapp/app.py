# c nuno example
import pickle
from gensim.corpora import Dictionary
from flask import Flask, request, render_template, jsonify
from topic_model import get_topics

# with open('/Users/lee/Documents/techniche/techniche/data/model_4.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('/Users/lee/Documents/techniche/techniche/data/model_4.pkl', 'rb') as f:
#     model = pickle.load(f)

with open('model_4.pkl', 'rb') as f_1:
    model = pickle.load(f_1)

with open('id_to_word_1000train.pkl', 'rb') as f_2:
    id_to_word = pickle.load(f_2)

app = Flask(__name__, static_url_path="")

@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html")


# @app.route("/output", methods=["GET", "POST"])
# def output():
#     """Return text from user input"""
#     data = request.get_json(force=True)
#     # every time the user_input identifier
#     print(data)
#     return jsonify(data["user_input"])


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


# def predict():
#      """Return a prediction of P(spam)."""
#      data = request.json
#      prediction = model.predict_proba([data['user_input']])
#      round_prediction = round(prediction[0][1], 2)
#      return jsonify({'probability': round_prediction})

# ME example
# import random
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# import pickle
# from flask import Flask, request, render_template, jsonify


# with open('spam_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# app = Flask(__name__, static_url_path="")

# @app.route('/')
# def index():
#     """Return the main page."""
#     return render_template(
#         'theme.html',
#         # words=['apple', 'orange']
#     )


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Return a prediction of P(spam)."""
#     data = request.json
#     prediction = model.predict_proba([data['user_input']])
#     round_prediction = round(prediction[0][1], 2)
#     return jsonify({'probability': round_prediction})


# glm extras
# import pickle
# from gensim.corpora import Dictionary
# # from topic_model import get_topics

# with open('/Users/lee/Documents/techniche/techniche/data/model_4.pkl', 'rb') as f:
#     model = pickle.load(f)

# app = Flask(__name__, static_url_path="")

# # if __name__ == '__main__':
# #     app.run(port=5000, debug=True)

# @app.route('/')
# def index():
#     """Return main page"""
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     """Return prediction"""
#     data = request.json
#     prediction = get_topics(id_to_word_1000train.doc2bow(text_input_1[data['user_input']]), model, k=15)
#     # return jsonify({'probability': prediction[0][1]})
#     return jsonify({'probability': prediction})

# @app.route("/output", methods=["GET", "POST"])
# def output():
#     """Return text from user input"""
#     data = request.get_json(force=True)
#     # every time the user_input identifier
#     print(data)
#     # cast dk to int
#     data["int"] = int(data["dk"])
#     data["mult"] = 10 * data["int"]
#     return jsonify(data["mult"])

# # below is direct from c nuno example
# # def output():
# #     """Retun text from user input"""
# #     data = request.get_json(force=True)
# #     # every time the user_input identifier
# #     print(data)
# #     return jsonify(data["user_input"])