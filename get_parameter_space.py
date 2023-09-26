
"""
Desc: The purpose of this script is to return the paramater search space for each ML algorithm.
      The space is defined in such a way as to be used with the hyperopt parameter tuning package.
	  
"""

from hyperopt import hp
import numpy as np

def return_parameter_space(algo):
    """Return parameter space for each algo."""

    space = dict()
    if algo == 'LR - no reg':
        space['solver'] = hp.choice('solver', ['lbfgs']) 
        space['penalty'] = hp.choice('penalty', ['none'])
        #space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif algo == 'LR - reg':
        space['solver'] = hp.choice('solver', ['liblinear']) 
        space['penalty'] = hp.choice('penalty', ['l1', 'l2']) 
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif algo == 'SVM':
        space['kernel'] = hp.choice('kernel', ['linear', 'rbf'])  
        space['gamma'] = hp.choice('gamma', ['scale', 'auto'])  
        space['C'] = hp.loguniform('C', np.log(0.00001), np.log(100))
    elif algo == 'NB':
    	space = {'var_smoothing': hp.loguniform('learning_rate', -20, 0)}
    elif algo == 'RF':
        #space['n_estimators'] = hp.quniform('n_estimators', range(50, 501, 10))  # Number of trees in the forest
        space['n_estimators'] = hp.choice('n_estimators', [50, 100, 200, 300, 400, 500])
        #space['max_depth']: hp.quniform('max_depth', 2, 20, 1)           # Maximum depth of each tree
        space['max_depth']: hp.choice('max_depth', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        space['min_samples_split']: hp.uniform('min_samples_split', 0.1, 1.0)  # Minimum samples required to split an internal node
        space['min_samples_split']: hp.choice('min_samples_split', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        space['min_samples_leaf']: hp.uniform('min_samples_leaf', 0.1, 0.5)    # Minimum samples required to be at a leaf node
        space['min_samples_leaf']: hp.choice('min_samples_leaf', [0.1, 0.2, 0.3, 0.4, 0.5])
        space['max_features']: hp.choice('max_features', ['auto', 'sqrt', 'log2', None])  # Number of features to consider when looking for the best split
        space['bootstrap']: hp.choice('bootstrap', [True, False]) 
    elif algo == 'XGB':
        #space['learning_rate'] = hp.loguniform('learning_rate', -5, 0),
        #space['max_depth'] = hp.quniform('max_depth', 3, 10, 1),
        #space['min_child_weight'] = hp.quniform('min_child_weight', 1, 10, 1), # Minimum sum of instance weight needed in a child
        #space['subsample'] = hp.uniform('subsample', 0.5, 1),  # Subsample ratio of the training instances
        #space['gamma'] = hp.uniform('gamma', 0, 1),  # Minimum loss reduction required to make a further partition
        #space['colsample_bytree'] = hp.uniform('colsample_bytree', 0.5, 1),  # Subsample ratio of columns when constructing a tree
        #space['colsample_bylevel'] = hp.uniform('colsample_bylevel', 0.5, 1),  # Subsample ratio of columns for each split level
        #space['colsample_bynode'] = hp.uniform('colsample_bynode', 0.5, 1),  # Subsample ratio of columns for each node
        #space['col_alpha'] = hp.loguniform('col_alpha', -6, 2),  # L1 regularization term on weights
        #space['reg_lambda'] = hp.loguniform('reg_lambda', -6, 2),  # L2 regularization term on weights
        #space['tree_method']: hp.choice('tree_method', ['auto', 'exact', 'approx', 'hist'])  # Tree construction method
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






