# test_predict_api.py
import json
from path import Path
from predict_api import app

DATA_DIR = Path(__file__).abspath().dirname()

def test_api():
    dataset_fname = DATA_DIR.joinpath('testdata_iris_v1.0.json')
    # Load all the test cases
    with open(dataset_fname) as f:
        test_data = json.load(f)

    with app.test_client() as client:
        for test_case in test_data:
            features = test_case['features']
            expected_response = test_case['expected_response']
            expected_status_code = test_case['expected_status_code']
            # Test client uses "query_string" instead of "params"
            response = client.get('/predict', query_string=features)
            # Check that we got "200 OK" back.
            assert response.status_code == expected_status_code
            # response.data returns a byte array, convert to a dict.
            assert json.loads(response.data) == expected_response
