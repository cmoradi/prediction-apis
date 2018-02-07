# test_predict_api.py
import json
from predict_api import app

def test_single_api_call():
    data = {'petal_length': 5.1, 'petal_width': 2.3,
            'sepal_length': 6.9, 'sepal_width': 3.1}
    expected_response = {
        "label": "virginica",
        "probabilities": {
            "setosa": 0.0,
            "versicolor": 0.2,
            "virginica": 0.8
        },
        "status": "complete"
    }

    with app.test_client() as client:
        # Test client uses "query_string" instead of "params"
        response = client.get('/predict', query_string=data)
        # Check that we got "200 OK" back.
        assert response.status_code == 200
        # response.data returns bytes, convert to a dict.
        assert json.loads(response.data) == expected_response
