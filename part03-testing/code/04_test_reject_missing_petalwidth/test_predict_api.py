# test_predict_api.py
import json
import pytest
from path import Path
from predict_api import app

# Find the directory where this script is.
# **ASSUMES THAT THE TEST DATASET FILES ARE HERE.
DATA_DIR = Path(__file__).abspath().dirname()

@pytest.mark.parametrize('filename',
                         ['testdata_iris_v1.0.json',
                          'testdata_iris_missing_v1.0.json'])
def test_api_from_file(filename):
    dataset_fname = DATA_DIR.joinpath(filename)
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


@pytest.mark.parametrize('data',
                         [{'petal_length': 5.1,
                           'sepal_length': 6.9,
                           'sepal_width': 3.1},
                          {'petal_length': 5.1,
                           'petal_width': 'junk',
                           'sepal_length': 6.9,
                           'sepal_width': 3.1},])
def test_reject_requests_missing_petal_width(data):
    expected_response = {
        "error_message": (
            "Record cannot be scored because petal_width "
            "is missing or has an unacceptable value."),
        "status": "error"
    }

    with app.test_client() as client:
        # Test client uses "query_string" instead of "params"
        response = client.get('/predict', query_string=data)
        # Check that we got "400 Bad Request" back.
        assert response.status_code == 400
        # response.data returns a byte array, convert to a dict.
        assert json.loads(response.data) == expected_response
