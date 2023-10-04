
"""
Desc: The purpose of this script is to return the paramater search space for each ML algorithm.
      The space is defined in such a way as to be used with the hyperopt parameter tuning package.
	  
"""

from hyperopt import hp
import numpy as np

def return_parameter_space(algo, cardinality):
    """Return parameter space for each algo."""

    space = dict()
    if algo == 'LR_no_reg':
        space['solver'] = hp.choice('solver', ['lbfgs']) 
        space['penalty'] = hp.choice('penalty', ['none'])
    elif (algo == 'LR_reg') & (cardinality == 'binary'):
        space['solver'] = hp.choice('solver', ['liblinear']) 
        space['penalty'] = hp.choice('penalty', ['l1', 'l2']) 
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif (algo == 'LR_reg') & (cardinality == 'multinomial'):
        space['solver'] = hp.choice('solver', ['saga']) 
        space['penalty'] = hp.choice('penalty', ['l1', 'l2']) 
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif algo == 'SVM':
        space['kernel'] = hp.choice('kernel', ['linear', 'rbf'])  
        space['gamma'] = hp.choice('gamma', ['scale', 'auto'])  
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif algo == 'NB':
    	space = {'var_smoothing': hp.loguniform('learning_rate', -20, 0)}
    elif algo == 'RF':
        space['n_estimators'] = hp.choice('n_estimators', [50, 100, 200, 300, 400, 500])
        space['max_depth'] = hp.choice('max_depth', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        space['min_samples_split'] = hp.choice('min_samples_split', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        space['min_samples_leaf'] = hp.choice('min_samples_leaf', [0.1, 0.2, 0.3, 0.4, 0.5])
        space['max_features'] = hp.choice('max_features', ['sqrt', 'log2', None])  # Number of features to consider when looking for the best split
        space['bootstrap'] = hp.choice('bootstrap', [True, False]) 
    elif algo == 'XGB':
        space = {'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500]), 
                 'booster': hp.choice('booster', ['gbtree']), # , 'dart'
                 'tree_method': hp.choice('tree_method', ['auto', 'hist', 'approx']), # , 'hist', 'approx', 'exact' 
                 'learning_rate': hp.loguniform('learning_rate', -5, 0),
                 'max_depth': hp.choice('max_depth', np.arange(3, 10, dtype=int)),
                 # hp.quniform('max_depth', 3, 10, 1) - causes XgBoostError as this passes floats whren int is required (hp.choice solves this)
                 'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                 'subsample': hp.uniform('subsample', 0.5, 1),
                 'gamma': hp.uniform('gamma', 0, 1),
                 'lambda': hp.uniform('lambda', 0, 5),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
                }

    return space






