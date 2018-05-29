import os
import sys
import numpy as np
import scipy

from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn import model_selection


algo = sys.argv[1]

dataset = fetch_california_housing()
X, y = dataset.data, dataset.target

cv = model_selection.ShuffleSplit(random_state=0)
clf = linear_model.Ridge()
param_dist = {'alpha': scipy.stats.expon()}


if algo == 'random':
    # run randomized search
    n_iter_search = 20
    random_search = model_selection.RandomizedSearchCV(
        clf, param_distributions=param_dist, cv=cv, n_iter=n_iter_search)
    random_search.fit(X, y)

    if not os.path.exists('data'):
        os.mkdir('data')
    np.save('data/score_%s.npy' % algo, random_search.cv_results_['mean_test_score'])
    np.save('data/parameters_%s.npy' % algo, random_search.cv_results_['params'])
