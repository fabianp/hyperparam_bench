# Twenty newsgroups hyperoptimization benchmark

import numpy as np
import sys

NJOBS = 2

# Data
from sklearn.datasets import fetch_20newsgroups
categories = None
newsgroups_train = fetch_20newsgroups(categories=categories,
                    subset='train', remove=('headers', 'footers', 'quotes'))


# Preproc
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


# Classifier
classifier_type = sys.argv[1]

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

if classifier_type == 'mnb':
    clf = MultinomialNB()
    param_grid = dict(classifier__alpha=np.logspace(-3, 3, 1000))
elif classifier_type == 'sgd':
    clf = SGDClassifier(loss='hinge', penalty='elasticnet')
    param_grid = dict(classifier__alpha=np.logspace(-4, 4, 1000),
                        classifier__l1_ratio=np.linspace(0, 1, 1000))


# Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline((('vectorizer', vectorizer),
                    ('classifier', clf)))


# Cross validation splits
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
cv = StratifiedShuffleSplit(random_state=0)


# Grid searches

search_type = sys.argv[2]

# ^ we might want to make this a distribution?



if search_type == 'random':
    from sklearn.model_selection import RandomizedSearchCV

    hyperparam_searcher = RandomizedSearchCV(
        pipeline, param_grid, n_iter=10, cv=cv,
        scoring='accuracy', verbose=1000, n_jobs=NJOBS)

    hyperparam_searcher.fit(newsgroups_train.data,
                            newsgroups_train.target)

elif search_type == 'grid':
    # This simply depends on where in the enumeration
    # the best value is. How can we properly compare this?
    pass
elif search_type == 'skopt':
    from skopt import BayesSearchCV

    hyperparam_searcher = BayesSearchCV(
        pipeline, param_grid, n_iter=10, cv=cv,
        scoring='accuracy', verbose=1000, n_jobs=NJOBS)

    hyperparam_searcher.fit(newsgroups_train.data,
                            newsgroups_train.target)
elif search_type == 'hyperopt':
    import hyperopt as hp
    from sklearn.model_selection import cross_val_score
    from sklearn.utils import clone
    def f_opt(**params):
        p = clone(pipeline)

        scores = cross_val_score(p, newsgroups_train.data,
                                            newsgroups_train.target,
                                            cv=cv, n_jobs=NJOBS)
        return scores.mean()
        


np.save('data/score_newsgroups_%s_%s.npy' % (classifier_type, search_type),
        hyperparam_searcher.cv_results_['mean_test_score'])
np.save('data/parameters_newsgroups_%s_%s.npy' % (classifier_type,
    search_type), 
        hyperparam_searcher.cv_results_['params'])




