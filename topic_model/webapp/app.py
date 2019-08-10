# c nuno example
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, static_url_path="")

@app.route("/")
def index():
    """Return the main page."""
    return render_template("index.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    """Retun text from user input"""
    data = request.get_json(force=True)
    # every time the user_input identifier
    print(data)
    return jsonify(data["user_input"])

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