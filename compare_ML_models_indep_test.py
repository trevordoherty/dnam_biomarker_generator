

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
from compare_ML_models import *
import hyperopt
from hyperopt import fmin, rand, tpe, hp, Trials, space_eval
import os
import pandas as pd
from pdb import set_trace
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import time
from xgboost import XGBClassifier
import warnings
hyperopt_rstate = np.random.RandomState(42)
warnings.filterwarnings("ignore")


def assess_ML_algorithm_indep_test(input_data, results_path, ml_algo_list):
    """Returns the best ML algroithm.

    Evaluated using a nested CV analysis on the training data set.
    The algorithm that achieves the best ROC AUC score is chosen.
    """

    def objective_lr(params):
        # Define the pipeline with a scaler and a classifier (RandomForestClassifier in this example)
        pipeline = Pipeline([('scaler', StandardScaler()),  # Add a scaler step
                             ('classifier', LogisticRegression(random_state=42, max_iter=100, n_jobs=-1, **params))])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')  # 5-fold cross-validation
        mean_auc = np.mean(scores)

        #pipeline.fit(X, y)
        #if len(set(y)) == 2:
        #    y_probas = pipeline.predict_proba(X_test)[:, 1]
        #    roc_auc = roc_auc_score(y_test_inner, y_probas)
        #elif len(set(y_train)) > 2:
        #    y_probas = model.predict_proba(X_test_inner)
        #    roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')          
        return -mean_auc


    best_algo = {'LR_no_reg': [], 'LR_reg': [], 'SVM': [], 'RF': [], 'XGB': [], 'NB': []}
    obj_fns = {'LR_reg_binary': objective_lr, 
               #'LR_reg_multinomial': objective_lr, 'LR_no_reg_binary': objective_lr,
               #'LR_no_reg_multinomial': objective_lr, 'SVM_binary': objective_svm, 'SVM_multinomial': objective_svm,
               #'RF_binary': objective_rf, 'RF_multinomial': objective_rf, 'XGB_binary': objective_xgb,
               #'XGB_multinomial': objective_xgb, 'NB_binary': objective_nb, 'NB_multinomial': objective_nb
               }
    results_list = []

    input_train = input_data[0]; input_test = input_data[1]
    y_indep = input_test['Label']; X_indep = input_test.drop(columns=['Label'])
    y = input_train['Label']; X = input_train.drop(columns=['Label'])
        
    for algo in ml_algo_list:
        start = time.time()
        # Set label cardinality key
        if len(set(y)) == 2:
            cardinality = 'binary'
        elif len(set(y)) > 2:
            cardinality = 'multinomial'
        
        space = return_parameter_space(algo, cardinality)
        trials = Trials()
        # Best hyperparameters
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
        y_pred = best_model.predict(X_indep)
        # Get predicted probabilities
        if cardinality == 'binary':
            y_probas = best_model.predict_proba(X_indep)[::, 1]
        elif cardinality == 'multinomial':
        	y_probas = best_model.predict_proba(X_indep)
        
        # ************PICK UP HERE - AMMEND PERFORMANCE METRICS FUNCTION ********
        # LOOK AT REFACTORING CODE TO REMOVE REPEATABILITY
        # CHECK WITH CHAPGPT    
        # Summarize the estimated performance of the model over nested CV outer test sets
        results = get_and_record_scores_indep(y_indep, y_pred, y_probas, cardinality)
        best_algo[algo] = results['auc']
        if not os.path.exists(results_path):            
        	os.makedirs(results_path)
        save_results_dictionary(results, results_path + 'results_' + str(algo) + '_hyperopt.pkl')        
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
    
