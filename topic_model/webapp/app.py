from flask import Flask, request, render_template, jsonify

with open('/Users/lee/Documents/techniche/techniche/data/model_lda_1.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, static_url_path="/www.techniche.app")

if __name__ == '__main__':
    app.run(port=5000, debug=True)

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    prediction = model.predict_proba([data['user_input']])
    return jsonify({'probability': prediction[0][1]})