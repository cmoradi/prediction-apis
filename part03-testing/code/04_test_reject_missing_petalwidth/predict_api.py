# Filename: predict_api.py
from pathlib import Path
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL_DIR = Path(__file__).parents[0]
MODEL_FILE = MODEL_DIR.joinpath('iris-rf-v1.0.pkl')
MODEL = joblib.load(MODEL_FILE)

MODEL_LABELS = ['setosa', 'versicolor', 'virginica']

HTTP_BAD_REQUEST = 400

@app.route('/predict')
def predict():
    sepal_length = request.args.get('sepal_length', default=5.8, type=float)
    sepal_width = request.args.get('sepal_width', default=3.0, type=float)
    petal_length = request.args.get('petal_length', default=3.9, type=float)
    petal_width = request.args.get('petal_width', default=None, type=float)

    if petal_width is None:
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because petal_width '
                   'is missing or has an unacceptable value.')
        response = jsonify(status='error',
                           error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Changed section.
    probabilities = MODEL.predict_proba(features)[0]
    label_index = probabilities.argmax()
    label = MODEL_LABELS[label_index]
    class_probabilities = dict(zip(MODEL_LABELS, probabilities.tolist()))
    return jsonify(status='complete', label=label,
                   probabilities=class_probabilities)


if __name__ == '__main__':
    app.run(debug=True)
