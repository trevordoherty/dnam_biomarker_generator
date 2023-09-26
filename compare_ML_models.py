

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
import hyperopt
from hyperopt import fmin, rand, tpe, hp, Trials, space_eval
import os
from pdb import set_trace
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
from xgboost import XGBClassifier
import warnings
hyperopt_rstate = np.random.RandomState(42)



def choose_ML_algorithm(input_data, results_path, ml_algo_list):
    """Returns the best ML algroithm.

    Evaluated using a nested CV analysis on the training data set.
    The algorithm that achieves the best ROC AUC score is chosen.
    """
    
    def objective_lr(params):
        model = LogisticRegression(random_state=42, max_iter=5000, **params) 
        model.fit(X_train, y_train)
        y_probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probas)
        return -roc_auc


    def objective_svm(params):
        model = SVC(random_state=42, probability=True, **params) 
        model.fit(X_train, y_train)
        y_probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probas)
        return -roc_auc


    def objective_nb(params):
        model = GaussianNB(**params) 
        model.fit(X_train, y_train)
        y_probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probas)
        return -roc_auc


    def objective_rf(params):
        params['n_estimators'] = int(params['n_estimators']) 
        model = RandomForestClassifier(random_state=42, **params) 
        model.fit(X_train, y_train)
        y_probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probas)
        return -roc_auc


    def objective_xgb(params):
        model = XGBClassifier(random_state=42, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric='auc', verbose=False, early_stopping_rounds=10)
        y_probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probas)
        return -roc_auc

    
    best_algo = {'LR - no reg': [], 'LR - reg': [], 'SVM': [], 'RF': [], 'XGB': [], 'NB': []}
    obj_fns = {'LR - reg': objective_lr, 'LR - no reg': objective_lr, 'SVM': objective_svm,
               'RF': objective_rf, 'XGB': objective_xgb, 'NB': objective_nb}
    for algo in ml_algo_list:
        start = time.time()
        results = {'auc': [], 'acc': [], 'sens': [], 'spec': [], 'prec': [],
                   'All test':[], 'All pred': [], 'All probas': []}
        outer_predictions = {'Fold predictions': [], 'Fold probabilities': [], 'Fold test': []}

        X = input_data.iloc[:, :-1]; y =input_data.iloc[:, -1]
        # configure the cross-validation procedure
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
        for train_ix, test_ix in cv_outer.split(X):
            # split data
            X_train, X_test = X.iloc[:, 0:500].iloc[train_ix, :], X.iloc[:, 0:500].iloc[test_ix, :]  # X.iloc[:, 0:500]
            y_train, y_test = y[train_ix], y[test_ix]
            
            if algo == 'XGB':
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                warnings.filterwarnings("ignore")
                
            if algo in ['LR - reg', 'LR - no reg', 'SVM', 'NB']: # Not scaled for RF or XGB
                # Scaling
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
		    
            space = return_parameter_space(algo)
            trials = Trials()

            best = fmin(fn=obj_fns[algo], space=space, algo=tpe.suggest, max_evals=100, trials=trials, rstate=hyperopt_rstate) 

            # Retrieve the best parameters
            best_params = space_eval(space, best)
            if algo in ['LR - reg', 'LR - no reg']:
                best_model = LogisticRegression(random_state=42, **best_params) # tree_method='hist', 
            elif algo == 'SVM':
            	best_model = SVC(random_state=42, probability=True, **best_params)
            elif algo == 'RF':
            	best_model = RandomForestClassifier(random_state=42, **best_params) # tree_method='hist'
            elif algo == 'NB':
                best_model = GaussianNB(**best_params)
            elif algo == 'XGB':
                best_model = XGBClassifier(random_state=42, **best_params)
            try:
                best_model.fit(X_train, y_train)
            except ValueError:
                continue

            # evaluate model on the hold out dataset
            y_pred = best_model.predict(X_test)
            # Get predicted probabilities
            y_probas = best_model.predict_proba(X_test)[::, 1] 
            outer_predictions['Fold predictions'].append((y_pred)); 
            outer_predictions['Fold probabilities'].append((y_probas))
            outer_predictions['Fold test'].append((y_test))
            
        best_algo[algo] = results['auc']   
        # Summarize the estimated performance of the model over nested CV outer test sets
        results = get_and_record_scores(outer_predictions, results)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        save_results_dictionary(results, results_path + 'results_' + str(algo) + '_hyperopt.pkl')        
        print("Duration for {}: {}".format(str(algo), time.time() - start))
    # Get max AUC and return best algo
    max_auc_algo = max(best_algo, key=best_algo.get)
    print('Algo with highest AUC: {}'.format(max_auc_algo))
    return max_auc_algo