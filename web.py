from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import joblib

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Load the trained model
model = joblib.load('y.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predictions = model.predict(data)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
