>>> from sklearn.externals import joblib
>>> from werkzeug.datastructures import MultiDict
>>> model = joblib.load('iris-rf-v1.0.pkl')
>>> r = MultiDict({"sepal_width": 2.6, "petal_length": 6.9, "petal_width": 2.3})
>>> model.predict(r)
{'label': 'virginica',
 'probabilities': {'setosa': 0.0, 'versicolor': 0.0, 'virginica': 1.0}}
>>> r2 = MultiDict({"sepal_width": 2.8, "petal_length": 4.8, "petal_width": 1.4})
>>> model.predict(r2)
{'label': 'versicolor',
 'probabilities': {'setosa': 0.0, 'versicolor': 0.8, 'virginica': 0.2}}
>>> r_missing = MultiDict({"sepal_width": 2.8, "petal_length": 4.8, "petal_length": 1.4})
>>> model.predict(r_missing)
---------------------------------------------------------------------------
ModelError                                Traceback (most recent call last)
<ipython-input-24-c1d8f5157495> in <module>()
----> 1 model.predict(r_missing)

/mnt/c/Users/chris/dev/prediction-apis/part04-decoupling/code/01_wrap_model/model_wrapper.py in predict(self, request_args)
     36
     37     def predict(self, request_args):
---> 38         features = self._prepare_features(request_args)
     39         try:
     40             probabilities = self._model.predict_proba([features])[0]

/mnt/c/Users/chris/dev/prediction-apis/part04-decoupling/code/01_wrap_model/model_wrapper.py in _prepare_features(self, request_args)
     31             message = ('Record cannot be scored because petal_width '
     32                        'is missing or has an unacceptable value.')
---> 33             raise ModelError(message)
     34
     35         return [sepal_length, sepal_width, petal_length, petal_width]

ModelError: Record cannot be scored because petal_width is missing or has an unacceptable value.
