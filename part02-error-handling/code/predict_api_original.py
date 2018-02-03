from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('iris-rf-v1.0.pkl')
MODEL_LABELS = ['setosa', 'versicolor', 'virginica']

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    sepal_length = request.args.get('sepal_length')
    sepal_width = request.args.get('sepal_width')
    petal_length = request.args.get('petal_length')
    petal_width = request.args.get('petal_width')

    # Our model expects a list of records
    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Use the model to predict the class
    label_index = MODEL.predict(features)
    # Retrieve the iris name that is associated with the predicted class
    label = MODEL_LABELS[label_index[0]]
    # Create and send a response to the API caller
    return jsonify(status='complete', label=label)

if __name__ == '__main__':
    app.run(debug=True)
