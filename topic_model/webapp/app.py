from flask import Flask, request, render_template, jsonify
import pickle

with open('/Users/lee/Documents/techniche/techniche/data/model_lda_1.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, static_url_path="")

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)

@app.route('/')
def index():
    """Return main page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return prediction"""
    data = request.json
    prediction = model.predict_proba([data['user_input']])
    return jsonify({'probability': prediction[0][1]})

@app.route("/output", methods=["GET", "POST"])
def output():
    """Return text from user input"""
    data = request.get_json(force=True)
    # every time the user_input identifier
    print(data)
    # cast dk to int
    data["int"] = int(data["dk"])
    data["mult"] = 10 * data["int"]
    return jsonify(data["mult"])