import json
import pytest

from predict_api_original import app as api_orig
from predict_api_catch_except_bad_requests import app as api_catch
from predict_api_reject_bad_requests import app as api_reject_bad
from predict_api_score_all import app as api_score_all


@pytest.fixture
def data():
    return {
        'good': {
            'petal_length': 5.1, 'petal_width': 2.3,
            'sepal_length': 6.9, 'sepal_width': 3.1},
        'missing': {
            'petal_length': 5.1, 'petal_width': 2.3,
            'sepal_width': 3.1},
        'bad': {
            'petal_length': 5.1, 'petal_width': 2.3,
            'sepal_length': 'junk', 'sepal_width': 3.1},

    }


def run_test(api, params, expected_status, expected_response):
    with api.test_client() as client:
        response = client.get('/predict', query_string=params)
        assert response.status_code == expected_status
        if response.status_code == 500:
            assert response.data == expected_response
        else:
            assert json.loads(response.data) == expected_response

def test_api_orig(data):
    error_msg = b'''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n<title>500 Internal Server Error</title>\n<h1>Internal Server Error</h1>\n<p>The server encountered an internal error and was unable to complete your request.  Either the server is overloaded or there is an error in the application.</p>\n'''
    run_test(api_orig, data['good'],
             200, dict(label='virginica', status='complete'))
    run_test(api_orig, data['missing'],
             500, error_msg)
    run_test(api_orig, data['bad'],
             500, error_msg)

def test_api_catch_except_bad_requests(data):
    error_msg = "Failed to score the model. Exception: Input contains NaN, infinity or a value too large for dtype('float32')."
    run_test(api_catch, data['good'],
             200, dict(label='virginica', status='complete'))
    run_test(api_catch, data['missing'],
             400, dict(status='error', error_message=error_msg))
    run_test(api_catch, data['bad'],
             400, dict(status='error', error_message=error_msg))

def test_api_catch_reject_bad(data):
    error_msg = 'Record cannot be scored because of missing or unacceptable values. All values must be present and of type float.'
    run_test(api_reject_bad, data['good'],
             200, dict(label='virginica', status='complete'))
    run_test(api_reject_bad, data['missing'],
             400, dict(status='error', error_message=error_msg))
    run_test(api_reject_bad, data['bad'],
             400, dict(status='error', error_message=error_msg))

def test_api_score_all(data):
    run_test(api_score_all, data['good'],
             200, dict(label='virginica', status='complete'))
    run_test(api_score_all, data['missing'],
             200, dict(label='virginica', status='complete'))
    run_test(api_score_all, data['bad'],
             200, dict(label='virginica', status='complete'))
