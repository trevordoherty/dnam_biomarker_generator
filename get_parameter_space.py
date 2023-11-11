
"""
Desc: The purpose of this script is to return the parameter search space for each ML algorithm.
      The space is defined in such a way as to be used with the hyperopt parameter tuning package.
	  
"""

from hyperopt import hp
import numpy as np
from scipy.stats import loguniform, uniform

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


def return_parameter_space_gs(algo, cardinality):
    """"Get parameter dict for gridsearch pipeline."""
    space = dict()
    if algo == 'LR_no_reg':
    	parameters = {'classifier__solver': ['lbfgs'],
    	              'classifier__penalty': ['none']}
    elif (algo == 'LR_reg') & (cardinality == 'binary'):
    	parameters = {'classifier__solver': ['liblinear'],
    	              'classifier__penalty': ['l1', 'l2'],
    	              'classifier__C': list(loguniform.rvs(0.00001, 100, size=100))} 
    elif (algo == 'LR_reg') & (cardinality == 'multinomial'):
        parameters = {'scaler': ['StandardScaler'],
    	              'classifier__solver': ['saga'],
    	              'classifier__penalty': ['l1', 'l2'],
    	              'classifier__C': list(loguniform.rvs(0.00001, 100, size=100))} 
    elif algo == 'SVM':
        parameters = {'classifier__kernel': ['linear', 'rbf'],
    	              'classifier__gamma': ['scale', 'auto'],
    	              'classifier__C': list(loguniform.rvs(0.00001, 100, size=20))}
    elif algo == 'NB':
    	parameters = {'classifier__var_smoothing': np.logspace(0,-9, num=100)}
    elif algo == 'RF':
        parameters = {'classifier__n_estimators': [50, 100, 200, 300, 400, 500],
    	              'classifier__max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    	              'classifier__min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    	              'classifier__min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
    	              'classifier__max_features': ['sqrt', 'log2', None],
    	              'classifier__bootstrap': [True, False]}
    elif algo == 'XGB':
        parameters = {'classifier__n_estimators': [50, 100, 200, 300, 400, 500],
    	              'classifier__booster': ['gbtree'],
    	              'classifier__tree_method': ['auto', 'hist', 'approx'],
    	              'classifier__learning_rate': [0.0001, 0.001, 0.01, 0.1],
    	              'classifier__max_depth': np.arange(3, 10, dtype=int),
    	              'classifier__min_child_weight': np.arange(1, 10, dtype=int),
                      'classifier__subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
                      'classifier__gamma': uniform(loc=0, scale=1),
                      'classifier__lambda': uniform(loc=0, scale=5),
                      'classifier__colsample_bytree': uniform(loc=0.5, scale=0.5)  
    	              }                  	
    return parameters



