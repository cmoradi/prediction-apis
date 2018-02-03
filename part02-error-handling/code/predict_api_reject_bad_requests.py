from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('iris-rf-v1.0.pkl')
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']

HTTP_BAD_REQUEST = 400


@app.route('/predict')
def predict():
    sepal_length = request.args.get('sepal_length', default=None, type=float)
    sepal_width = request.args.get('sepal_width', default=None, type=float)
    petal_length = request.args.get('petal_length', default=None, type=float)
    petal_width = request.args.get('petal_width', default=None, type=float)

    # Reject request that have bad or missing values.
    if (sepal_length is None or sepal_width is None
        or petal_length is None or petal_width is None):
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because of '
                   'missing or unacceptable values. '
                   'All values must be present and of type float.')
        response = jsonify(status='error',
                           error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    label_index = MODEL.predict(features)
    label = MODEL_LABELS[label_index[0]]
    return jsonify(status='complete', label=label)

if __name__ == '__main__':
    app.run(debug=True)
