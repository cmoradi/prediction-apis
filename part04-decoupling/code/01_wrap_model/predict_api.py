# Filename: predict_api.py
from pathlib import Path
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL_DIR = Path(__file__).parents[0]
MODEL_FILE = MODEL_DIR.joinpath('iris-rf-v1.0.pkl')
MODEL = joblib.load(MODEL_FILE)

HTTP_BAD_REQUEST = 400

@app.route('/predict')
def predict():
    try:
        result = MODEL.predict(request.args)
    except MODEL.ModelError as err:
        response = jsonify(status='error',
                           error_message=str(err))
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    return jsonify(status='complete', **result)

if __name__ == '__main__':
    app.run(debug=True)
