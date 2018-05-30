# Twenty newsgroups hyperoptimization benchmark

import numpy as np
import sys

NJOBS = -1

dataset_name = sys.argv[1]
if dataset_name == 'newsgroups':
    from newsgroups_data import get_newsgroups_info as get_info
elif dataset_name == 'housing':
    from housing_data import get_housing_info as get_info
else:
    raise NotImplementedError(f"We don't have dataset {dataset_name} yet")


for seed in range(10):

    info = get_info(seed)
    Xtrain, ytrain, cv, classifiers, scoring = (info[item]
                for item in ('Xtrain', 'ytrain', 'cv', 'classifiers', 'scoring'))

    classifier_type = sys.argv[2]

#    if classifier_type == 'mnb':
#       pipeline, param_grids = classifiers[1]
#    elif classifier_type == 'sgd':
#       pipeline, param_grids = classifiers[0]

    pipeline, param_grids = classifiers[classifier_type]

    param_grid = param_grids['hyperparams_random_search']
    param_grid_skopt = param_grids['hyperparams_skopt']
    param_grid_hyperopt = param_grids['hyperparams_hyperopt']

    search_type = sys.argv[3]


    if search_type == 'random':
        from sklearn.model_selection import RandomizedSearchCV

        hyperparam_searcher = RandomizedSearchCV(
            pipeline, param_grid, n_iter=100, cv=cv,
            scoring=scoring,
            verbose=1, n_jobs=NJOBS, random_state=seed)

        hyperparam_searcher.fit(Xtrain, ytrain)

    elif search_type == 'grid':
        # This simply depends on where in the enumeration
        # the best value is. How can we properly compare this?
        raise NotImplementedError("Currently no grid search")
    elif search_type == 'skopt':
        from skopt import BayesSearchCV

        hyperparam_searcher = BayesSearchCV(
            pipeline, param_grid_skopt, n_iter=100, cv=cv,
            scoring=scoring, verbose=1, n_jobs=NJOBS, random_state=seed)

        hyperparam_searcher.fit(Xtrain, ytrain)

    elif search_type == 'hyperopt':
        import hyperopt as hp
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        from hyperopt import fmin, tpe, hp, STATUS_OK

        param_names = list(param_grid_hyperopt.keys())
        param_values = [param_grid_hyperopt[key] for key in param_names]

        inputs = []
        outputs = []
        def f_opt(param_tuple, param_names=param_names):
            params = dict(zip(param_names, param_tuple))
            p = clone(pipeline)
            p.set_params(**params)
            scores = cross_val_score(p, Xtrain, ytrain, cv=cv, scoring=scoring, n_jobs=NJOBS)
            inputs.append(param_tuple)
            output = scores.mean()
            outputs.append(output)
            return {'loss': output, 'status': STATUS_OK}

        best = fmin(f_opt, space=param_values, algo=tpe.suggest, max_evals=100)
        # Do some hacky stuff to be able to use the save code below :D
        hyperparam_searcher = lambda x: None
        hyperparam_searcher.cv_results_ = dict(mean_test_score=np.array(outputs),
                             params=[dict(zip(param_names, values))
                                 for values in inputs])

    np.save('data/score_%s_%s_%s_%s.npy' % (dataset_name,
                                            classifier_type,
                                                    search_type,
                                                    seed),
            hyperparam_searcher.cv_results_['mean_test_score'])
    np.save('data/parameters_%s_%s_%s_%s.npy' % (dataset_name,
                                                classifier_type,
                                                        search_type,
                                                        seed),
            hyperparam_searcher.cv_results_['params'])

