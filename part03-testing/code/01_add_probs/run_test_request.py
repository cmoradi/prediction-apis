import requests

data = {'petal_length': 5.1, 'petal_width': 2.3,
        'sepal_length': 6.9, 'sepal_width': 3.1}

response = requests.get('http://127.0.0.1:5000/predict', params=data)
print('Status code: {}'.format(response.status_code))
print('Payload:\n{}'.format(response.text))

# Response should be something like this:
# Status code: 200
# Payload:
# {
#   "label": "virginica",
#   "probabilities": {
#     "setosa": 0.0,
#     "versicolor": 0.2,
#     "virginica": 0.8
#   },
#   "status": "complete"
# }
