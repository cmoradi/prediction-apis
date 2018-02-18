import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from model_wrapper import ModelWrapper

MODEL_VERSION = '1.0'

def prep_test_cases(all_features, all_probs, feature_names, target_names):
    all_test_cases = []
    for feat_vec, prob_vec in zip(all_features, all_probs):
        # Drop features that have value == None
        # Here's the change >>>>>>
        feat_dict = {name: val for name, val
                     in zip(feature_names, feat_vec)
                     if not val is None}
        prob_dict = dict(zip(target_names, prob_vec))
        expected_label = target_names[prob_vec.argmax()]
        expected_response = dict(label=expected_label,
                                 probabilities=prob_dict,
                                 status='complete')
        test_case = dict(features=feat_dict,
                         expected_status_code=200,
                         expected_response=expected_response)
        all_test_cases.append(test_case)
    return all_test_cases

def main():
    # Grab the dataset from scikit-learn
    data = datasets.load_iris()
    X = data['data']
    y = data['target']
    target_names = data['target_names']
    feature_names = [f.replace(' (cm)', '').replace(' ', '_')
                     for f in data.feature_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    # Build and train the model
    model = RandomForestClassifier(random_state=101)
    model.fit(X_train, y_train)
    print("Score on the training set is: {:2}"
          .format(model.score(X_train, y_train)))
    print("Score on the test set is: {:.2}"
          .format(model.score(X_test, y_test)))

    # CHANGES HERE >>>>>>>>>>
    X_mean = X_train.mean(axis=0).round(1)
    feature_defaults = dict(zip(feature_names, X_mean.tolist()))
    wrapped = ModelWrapper(model_name='iris-rf',
                           model_version=MODEL_VERSION,
                           model_object=model,
                           class_labels=target_names.tolist(),
                           feature_defaults=feature_defaults)

    # Save the model
    model_filename = 'iris-rf-v{}.pkl'.format(MODEL_VERSION)
    print("Saving model to {}...".format(model_filename))
    joblib.dump(wrapped, model_filename)

    # ***** Generate test data *****
    print('Generating test data...')
    all_probs = model.predict_proba(X_test)
    all_test_cases = prep_test_cases(X_test,
                                     all_probs,
                                     feature_names,
                                     target_names)
    test_data_fname = 'testdata_iris_v{}.json'.format(MODEL_VERSION)
    with open(test_data_fname, 'w') as fout:
        json.dump(all_test_cases, fout)

    # ***** Generate test data (Missing) *****
    print('Generating test data with missing values...')
    missing_grps = [(0,), (1,), (2,),
                    (0, 1), (0, 2), (1, 2),
                    (0, 1, 2)]
    X_mean = X_train.mean(axis=0).round(1)
    all_features = []
    all_probs = []
    for missing_cols in missing_grps:
        X_missing = X_test.copy().astype('object')
        X_scored = X_test.copy()
        for col in missing_cols:
            X_missing[:, col] = None
            X_scored[:, col] = X_mean[col]
        # Use the imputed one to find expected probabilities
        all_probs.extend(model.predict_proba(X_scored))
        all_features.extend(X_missing)

    all_test_cases_missing = prep_test_cases(all_features,
                                             all_probs,
                                             feature_names,
                                             target_names)

    test_data_fname = 'testdata_iris_missing_v{}.json'.format(MODEL_VERSION)
    with open(test_data_fname, 'w') as fout:
        json.dump(all_test_cases_missing, fout)

if __name__ == '__main__':
    main()
