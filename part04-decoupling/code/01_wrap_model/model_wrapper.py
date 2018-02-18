class ModelError(Exception):
    pass

class ModelWrapper:

    def __init__(self, *, model_name, model_version, model_object,
                 class_labels, feature_defaults):
        self.model_name = model_name
        self.model_version = model_version
        self._model = model_object
        self.class_labels = class_labels
        self.feature_defaults = feature_defaults
        self.ModelError = ModelError

    def predict(self, request_args):
        features = self._prepare_features(request_args)
        try:
            probabilities = self._model.predict_proba([features])[0]
            label_index = probabilities.argmax()
            label = self.class_labels[label_index]
            class_probabilities = dict(zip(self.class_labels, probabilities.tolist()))
        except Exception as err:
            # Here we should log the error, but we'll cover that later.
            msg_template = ('Failed to score model {name} v{version} '
                            'with features: {features}')
            message = msg_template.format(name=self.model_name,
                                          version=self.model_version,
                                          features=features)
            raise ModelError(message)

        result = dict(label=label, probabilities=class_probabilities)
        return result

    def _prepare_features(self, request_args):
        sepal_length = request_args.get('sepal_length',
                                        default=self.feature_defaults['sepal_length'],
                                        type=float)
        sepal_width = request_args.get('sepal_width',
                                        default=self.feature_defaults['sepal_width'],
                                        type=float)
        petal_length = request_args.get('petal_length',
                                        default=self.feature_defaults['petal_length'],
                                        type=float)
        # For this feature, we don't want to impute.
        petal_width = request_args.get('petal_width',
                                        default=None,
                                        type=float)

        if petal_width is None:
            # Create a message that the API will return.
            message = ('Record cannot be scored because petal_width '
                       'is missing or has an unacceptable value.')
            raise ModelError(message)

        return [sepal_length, sepal_width, petal_length, petal_width]
