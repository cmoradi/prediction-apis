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

    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    try:
        label_index = MODEL.predict(features)
    except Exception as err:
        message = ('Failed to score the model. Exception: {}'.format(err))
        response = jsonify(status='error', error_message=message)
        response.status_code = HTTP_BAD_REQUEST
        return response

    label = MODEL_LABELS[label_index[0]]
    return jsonify(status='complete', label=label)

if __name__ == '__main__':
    app.run(debug=True)
