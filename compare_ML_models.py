

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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import time
from xgboost import XGBClassifier
import warnings
hyperopt_rstate = np.random.RandomState(42)
warnings.filterwarnings("ignore")


def assess_ML_algorithm_nested_cv(input_data, results_path, ml_algo_list):
    """Returns the best ML algroithm.

    Evaluated using a nested CV analysis on the training data set.
    The algorithm that achieves the best ROC AUC score is chosen.
    """
    
    def standardise(X_train_inner, X_test_inner):
        """Apply standardisation within the inner train/test split."""
        # Scaling
        sc = StandardScaler()
        X_train_inner = sc.fit_transform(X_train_inner)
        X_test_inner = sc.transform(X_test_inner)
        return X_train_inner, X_test_inner


    def objective_lr(params):
        #auc_scores = []
        #cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        #for train_index, test_index in cv.split(X_train, y_train):
        #    X_train_inner, X_test_inner = X_train[train_index], X_train[test_index] 
        #    y_train_inner, y_test_inner = y[train_index], y[test_index] 
        X_train_inner, X_test_inner = standardise(X_train_inner, X_test_inner)
        model = LogisticRegression(random_state=42, max_iter=100, n_jobs=-1, **params) #, max_iter=5000
        model.fit(X_train_inner, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')          
        #auc_scores.append(roc_auc)
        #mean_auc = np.mean(auc_scores)
        return -roc_auc


    def objective_svm(params):
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train_inner, X_test_inner = standardise(X_train_inner, X_test_inner)
        model = SVC(random_state=42, probability=True, **params) 
        model.fit(X_train_inner, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc


    def objective_nb(params):
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train_inner, X_test_inner = standardise(X_train_inner, X_test_inner)
        model = GaussianNB(**params) 
        model.fit(X_train_inner, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc


    def objective_rf(params):
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        params['n_estimators'] = int(params['n_estimators']) 
        model = RandomForestClassifier(random_state=42, **params) 
        model.fit(X_train_inner, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc


    def objective_xgb(params):
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train_inner, X_val, y_train_inner, y_val = \
            train_test_split(X_train_inner, y_train_inner, test_size=0.2, random_state=42)
        model = XGBClassifier(random_state=42, **params)
        model.fit(X_train_inner, y_train_inner, eval_set=[(X_val, y_val)],
              eval_metric='auc', verbose=False, early_stopping_rounds=10)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc

    
    best_algo = {'LR_no_reg': [], 'LR_reg': [], 'SVM': [], 'RF': [], 'XGB': [], 'NB': []}
    obj_fns = {'LR_reg_binary': objective_lr, 'LR_reg_multinomial': objective_lr, 'LR_no_reg_binary': objective_lr,
               'LR_no_reg_multinomial': objective_lr, 'SVM_binary': objective_svm, 'SVM_multinomial': objective_svm,
               'RF_binary': objective_rf, 'RF_multinomial': objective_rf, 'XGB_binary': objective_xgb,
               'XGB_multinomial': objective_xgb, 'NB_binary': objective_nb, 'NB_multinomial': objective_nb
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
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
            
    for algo in ml_algo_list:
        start = time.time()
        # Set label cardinality key
        if len(set(y)) == 2:
            cardinality = 'binary'
        elif len(set(y)) > 2:
            cardinality = 'multinomial'
        outer_predictions = {'Fold predictions': [], 'Fold probabilities': [], 'Fold test': []}
        
        for train_ix, test_ix in cv_outer.split(X):
            # split data
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]  # X.iloc[:, 0:500]
            y_train, y_test = y[train_ix], y[test_ix]

            space = return_parameter_space(algo, cardinality)
            trials = Trials()
            
            best = fmin(fn=obj_fns[algo + '_' + cardinality], space=space,
                algo=partial(mix.suggest, 
                             p_suggest=[(.1, rand.suggest), (.2, anneal.suggest), (.7, tpe.suggest),]),
                             max_evals=100, trials=trials, rstate=hyperopt_rstate)

            # Retrieve the best parameters
            best_params = space_eval(space, best)
            if algo in ['LR_reg', 'LR_no_reg']:
                best_model = LogisticRegression(random_state=42, n_jobs=-1, **best_params) # tree_method='hist', 
            elif algo == 'SVM':
            	best_model = SVC(random_state=42, probability=True, **best_params)
            elif algo == 'RF':
            	best_model = RandomForestClassifier(random_state=42, **best_params) # tree_method='hist'
            elif algo == 'NB':
                best_model = GaussianNB(**best_params)
            elif algo == 'XGB':
                best_model = XGBClassifier(random_state=42, **best_params)
            best_model.fit(X_train, y_train)

            # evaluate model on the hold out dataset
            y_pred = best_model.predict(X_test)
            # Get predicted probabilities
            if cardinality == 'binary':
                y_probas = best_model.predict_proba(X_test)[::, 1]
            elif cardinality == 'multinomial':
            	y_probas = best_model.predict_proba(X_test)
            outer_predictions['Fold predictions'].append((y_pred)); 
            outer_predictions['Fold probabilities'].append((y_probas))
            outer_predictions['Fold test'].append((y_test))

        # Summarize the estimated performance of the model over nested CV outer test sets
        results = get_and_record_scores(outer_predictions, cardinality)
        best_algo[algo] = results['auc']
        if not os.path.exists(results_path):
        	os.makedirs(results_path)
        save_results_dictionary(results, results_path + 'results_' + str(algo) + '.pkl')
        print("Duration for {}: {}".format(str(algo), time.time() - start))
        # Remove dictionary items not needed for results output table
        for k in ['All probas', 'All pred', 'All test']: results.pop(k, None)
        results['Algorithm'] = algo; results_list.append(results)


    # Get max AUC and return best algo
    best_algo = {k: v for k, v in best_algo.items() if v}
    max_auc_algo = max(best_algo, key=best_algo.get)
    results_df = pd.DataFrame(results_list)
    col1 = results_df.pop('Algorithm'); results_df.insert(0, 'Algorithm', col1)
    print("Results table for nested CV ...")
    print(tabulate(results_df, headers='keys', tablefmt='psql'))
    return max_auc_algo

