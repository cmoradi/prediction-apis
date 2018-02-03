from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('iris-rf-v1.0.pkl')
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']

HTTP_BAD_REQUEST = 400


@app.route('/predict')
def predict():
    # Using hardcoded numbers here, but these should really be variables.
    sepal_length = request.args.get('sepal_length', default=5.8, type=float)
    sepal_width = request.args.get('sepal_width', default=3.0, type=float)
    petal_length = request.args.get('petal_length', default=3.9, type=float)
    petal_width = request.args.get('petal_width', default=1.2, type=float)

    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    label_index = MODEL.predict(features)
    label = MODEL_LABELS[label_index[0]]
    return jsonify(status='complete', label=label)

if __name__ == '__main__':
    app.run(debug=True)
