

"""
Desc: This script contains functionality that compares multiple ML models.
	  The models for comparison can be passed as a list argument when running the code.
      For example, if yu only want to compare a logistic regression and a gradient boosted
      model, you would pass ['LR', XGB'].
      Alternatively, no passing an argument will run all ML algorithms in package.

      The best model is selected based on AUC score from a nested CV analysis on the training
      data set. Then, this ML algroithm is used with the various feature selection approaches
      to find the best method. The final model is trained on the full training data using the
      best feature reduction and ML algorithm.

      This final trained model can then be used for prediction on new data sets.
"""

from assess_performance import *
from get_parameter_space import *
from functools import partial
import hyperopt
from hyperopt import fmin, rand, tpe, hp, Trials, space_eval, anneal, mix
import os
import pandas as pd
from pdb import set_trace
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import time
from xgboost import XGBClassifier
import warnings
hyperopt_rstate = np.random.RandomState(42)
warnings.filterwarnings("ignore")
seed_value = 42
np.random.seed(seed_value)


def assess_ML_algorithm_nested_cv_rs(input_data, inputs_path, results_path, ml_algo_list):
    """Returns the best ML algroithm.

    Evaluated using a nested CV analysis on the training data set.
    The algorithm that achieves the best ROC AUC score is chosen.
    """
    
    best_algo = {'LR_no_reg': [], 'LR_reg': [], 'SVM': [], 'RF': [], 'XGB': [], 'NB': []}
    pipes = {'LR_reg': Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression(random_state=42, n_jobs=-1))]),
             'LR_no_reg': Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression(random_state=42, n_jobs=-1))]),
             'SVM': Pipeline([('scaler', StandardScaler()),('classifier', SVC(random_state=42, probability=True))]),
             'NB': Pipeline([('scaler', StandardScaler()),('classifier', GaussianNB())]),
             'RF': Pipeline([('classifier', RandomForestClassifier(random_state=42))]),
             'XGB': Pipeline([('classifier', XGBClassifier(random_state=42))])
             }

    results_list = []
    if len(input_data) == 1:
        input_train = input_data[0]
        #input_train = input_train.sample(frac=0.5)
    elif len(input_data) == 2:
        input_train = input_data[0]; input_test = input_data[1]
        y_indep = input_test['Label']; X_indep = input_test.drop(columns=['Label'])
    y = input_train['Label']; X = input_train.drop(columns=['Label'])
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            
    for algo in ml_algo_list:
        start = time.time()
        # Set label cardinality key
        if len(set(y)) == 2:
            cardinality = 'binary'
        elif len(set(y)) > 2:
            cardinality = 'multinomial'
        outer_predictions = {'Fold predictions': [], 'Fold probabilities': [], 'Fold test': []}
        
        for train_ix, test_ix in cv_outer.split(X, y):
            # split data
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]  # X.iloc[:, 0:500]
            y_train, y_test = y[train_ix], y[test_ix]


            # Configure the inner cross-validation
            cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
            parameters = return_parameter_space_gs(algo, cardinality)
            if algo == "XGB":
                X_train_es, X_val, y_train_es, y_val = \
                    train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                fit_params = {"eval_set": [(X_val, y_val)], "eval_metric": 'auc', "early_stopping_rounds": 20}
                grid = RandomizedSearchCV(pipes[algo],
                                          parameters,
                                          cv=cv_inner,
                                          n_iter=50,
                                          random_state=42).fit(X_train_es,
                                                               y_train_es)
                best_params = grid.best_params_
                best_params = {key[12:]:value for key, value in best_params.items()}
                
                model = XGBClassifier(random_state=42, **best_params)
                model.fit(X_train_es, y_train_es, eval_metric='auc',
                          eval_set=[(X_val, y_val)], early_stopping_rounds=20)
                y_pred = model.predict(X_test)
                # Get predicted probabilities
                if cardinality == 'binary':
                    y_probas = model.predict_proba(X_test)[::, 1]
                elif cardinality == 'multinomial':
                    y_probas = model.predict_proba(X_test)
            else:
                model = RandomizedSearchCV(pipes[algo], parameters, cv=cv_inner, n_iter=50, random_state=42).fit(X_train, y_train)
                y_pred = model.best_estimator_.predict(X_test)
                # Get predicted probabilities
                if cardinality == 'binary':
                    y_probas = model.best_estimator_.predict_proba(X_test)[::, 1]
                elif cardinality == 'multinomial':
                    y_probas = model.best_estimator_.predict_proba(X_test)
            
            outer_predictions['Fold predictions'].append((y_pred)); 
            outer_predictions['Fold probabilities'].append((y_probas))
            outer_predictions['Fold test'].append((y_test))

        # Summarize the estimated performance of the model over nested CV outer test sets
        results = get_and_record_scores(outer_predictions, cardinality)
        best_algo[algo] = results['auc']
        if not os.path.exists(results_path):
        	os.makedirs(results_path)
        save_results_dictionary(results, results_path + 'results_rs_' + str(algo) + '.pkl')
        print("Duration for {}: {}".format(str(algo), time.time() - start))
        # Remove dictionary items not needed for results output table
        for k in ['All probas', 'All pred', 'All test']: results.pop(k, None)
        results['Algorithm'] = algo; results_list.append(results)


    # Get max AUC and return best algo
    best_algo = {k: v for k, v in best_algo.items() if v}
    max_auc_algo = max(best_algo, key=best_algo.get)
    
    results_df = pd.DataFrame(results_list); 
    results_df['Input'] = inputs_path[0][inputs_path[0].rfind("/") + 1:inputs_path[0].find(".", inputs_path[0].rfind("/"))]
    results_df['Tuning'] = 'RandomizedSearchCV'
    col1 = results_df.pop('Algorithm'); results_df.insert(0, 'Algorithm', col1)
    print("Results table for nested CV ...")
    print(tabulate(results_df, headers='keys', tablefmt='psql'))
    return max_auc_algo

