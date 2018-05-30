from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Ridge
from utils import loguniform

from hyperopt import hp

def _get_data():
     dataset = fetch_california_housing()
     X, y = dataset.data, dataset.target
     return X, y


def _get_cv(seed):
    cv = ShuffleSplit(random_state=seed)
    return cv


def _get_ridge():
    est = Ridge()
    hyperparams = dict(
            hyperparams_random_search=dict(
                alpha=loguniform(1e-3, 1e5)
                ),
            hyperparams_skopt=dict(
                alpha=(1e-3, 1e5, 'log-uniform')
                ),
            hyperparams_hyperopt=dict(
                alpha=hp.loguniform(
                    'alpha', -3, 5)
                )
            )
    return est, hyperparams


def get_housing_info(seed):
    Xtrain, ytrain = _get_data()
    cv = _get_cv(seed)
    ridge = _get_ridge()

    output = dict(Xtrain=Xtrain, ytrain=ytrain,
                    cv=cv, classifiers=dict(ridge=ridge), scoring='r2')
    return output



