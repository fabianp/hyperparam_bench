import numpy as np
from scipy import stats
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit

from hyperopt import hp
from scipy.stats import uniform
from utils import loguniform

__all__ = ['get_newsgroups_info']

def _get_data():
    categories = None
    newsgroups_train = fetch_20newsgroups(categories=categories,
                    subset='train', remove=('headers', 'footers', 'quotes'))
    return newsgroups_train.data, newsgroups_train.target


def _get_cv(seed):
    cv = StratifiedShuffleSplit(random_state=seed)
    return cv


def _get_sgd_clf():
    vectorizer = TfidfVectorizer()
    clf = SGDClassifier(loss='hinge', penalty='elasticnet')
    pipeline = Pipeline((('vectorizer', vectorizer),
                    ('classifier', clf)))

    hyperparams = dict(
        hyperparams_skopt=dict(
                    classifier__alpha=(1e-3, 1e5, 'log-uniform'),
                    classifier__l1_ratio=(0, 1, 'uniform')),
        hyperparams_hyperopt=dict(
                    classifier__alpha=hp.loguniform('alpha', -3, 5),
                    classifier__l1_ratio=hp.uniform('l1_ratio', 0, 1)),
        hyperparams_random_search=dict(
                    classifier__alpha=loguniform(1e-3, 1e5),
                    classifier__l1_ratio=uniform()
            ))
    return pipeline, hyperparams


def _get_mnb_clf():
    vectorizer = TfidfVectorizer()
    clf = MultinomialNB()
    pipeline = Pipeline((('vectorizer', vectorizer),
                    ('classifier', clf)))

    hyperparams = dict(
        hyperparams_skopt=dict(
                    classifier__alpha=(1e-3, 1e5, 'log-uniform')),
        hyperparams_hyperopt=dict(
                    classifier__alpha=hp.loguniform('alpha', -3, 5)),
        hyperparams_random_search=dict(
                    classifier__alpha=loguniform(1e-3, 1e5),
            ))
    return pipeline, hyperparams


def get_newsgroups_info(seed):

    Xtrain, ytrain = _get_data()
    cv = _get_cv(seed)
    sgd = _get_sgd_clf()
    mnb = _get_mnb_clf()

    output = dict(Xtrain=Xtrain,
                    ytrain=ytrain,
                    cv=cv,
                    classifiers=[sgd, mnb])
    return output


