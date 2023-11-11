

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
from feature_reduction_methods import *
from get_parameter_space import *
from functools import partial
import hyperopt
from hyperopt import fmin, rand, tpe, hp, Trials, space_eval, anneal, mix
import matplotlib.pyplot as plt
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

'''
def standardise(X_train_inner, X_test_inner):
    """Apply standardisation within the inner train/test split."""
    # Scaling
    sc = StandardScaler()
    X_train_inner = sc.fit_transform(X_train_inner)
    X_test_inner = sc.transform(X_test_inner)
    return X_train_inner, X_test_inner
'''

def assess_feature_selection_nested_cv(input_data, results_path, ml_algo_list):
    """Returns the best ML algroithm.

    Evaluated using a nested CV analysis on the training data set.
    The algorithm that achieves the best ROC AUC score is chosen.
    """


    def objective_lr(params):
        model = LogisticRegression(random_state=42, max_iter=100, n_jobs=-1, **params) #, max_iter=5000
        model.fit(X_train_inner_sub, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner_sub)[:, 1]
            try:
                roc_auc = roc_auc_score(y_test_inner, y_probas)
            except:
                set_trace()
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner_sub)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')          
        return -roc_auc


    def objective_svm(params):
        model = SVC(random_state=42, probability=True, **params) 
        model.fit(X_train_inner_sub, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner_sub)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner_sub)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc


    def objective_nb(params):
        model = GaussianNB(**params) 
        model.fit(X_train_inner_sub, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner_sub)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner_sub)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc


    def objective_rf(params):
        params['n_estimators'] = int(params['n_estimators']) 
        model = RandomForestClassifier(random_state=42, **params) 
        model.fit(X_train_inner_sub, y_train_inner)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner_sub)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner_sub)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc


    def objective_xgb(params):
        X_train_inner_sub_es, X_val, y_train_inner_es, y_val = \
            train_test_split(X_train_inner_sub, y_train_inner, test_size=0.2, random_state=42)
        model = XGBClassifier(random_state=42, **params)
        model.fit(X_train_inner_sub_es, y_train_inner_es, eval_set=[(X_val, y_val)],
              eval_metric='auc', verbose=False, early_stopping_rounds=10)
        if len(set(y_train)) == 2:
            y_probas = model.predict_proba(X_test_inner_sub)[:, 1]
            roc_auc = roc_auc_score(y_test_inner, y_probas)
        elif len(set(y_train)) > 2:
            y_probas = model.predict_proba(X_test_inner_sub)
            roc_auc = roc_auc_score(y_test_inner, y_probas, multi_class= 'ovr', average='micro')
        return -roc_auc

    
    feat_max = 500


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
    cv_outer = KFold(n_splits=2, shuffle=True, random_state=1)
    
    methods = ["AE", "PCA"] # ,  "Variance", "MI", "ANOVA", "Random Forest" "PCA", 
    for method in methods:
        reduced_features = get_reduced_features(cv_outer, X, y, method)        
        if method in ["Random Forest", "Variance", "MI", "ANOVA"]:
            subsets = list(range(100, feat_max + 1, 100))
            outer_predictions = {method: {key: {} for key in range(100, feat_max + 1, 100)}}
        elif method in ["PCA", "AE"]:
            subsets = [reduced_features[0][2].shape[1]]
            outer_predictions = {'PCA': {reduced_features[0][2].shape[1]: {}},
                                 'AE': {reduced_features[0][2].shape[1]: {}}}
        for algo in ml_algo_list:
            start = time.time()
            # Set label cardinality key
            if len(set(y)) == 2:
                cardinality = 'binary'
            elif len(set(y)) > 2:
                cardinality = 'multinomial'
            
            for subset in subsets:
                outer_predictions[method][subset]['Fold predictions'] = [];
                outer_predictions[method][subset]['Fold probabilities'] = [];
                outer_predictions[method][subset]['Fold test'] = [];
                #for train_ix, test_ix in cv_outer.split(X):
                for fold_index, (train_ix, test_ix) in enumerate(cv_outer.split(X)):
                    # split data
                    X_train, X_test = X.iloc[:, 0:feat_max].iloc[train_ix, :], X.iloc[:, 0:feat_max].iloc[test_ix, :]  # X.iloc[:, 0:500]
                    y_train, y_test = y[train_ix], y[test_ix]
                    X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
                        train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    
                    if method in ["Random Forest", "Variance", "MI", "ANOVA"]:
                        # At this point, get the feature ranking on X_train_inner and X_train
                        sub_inner = reduced_features[fold_index][1].index[0:subset]
                        sub = reduced_features[fold_index][2].index[0:subset]
                    
                        X_train_inner_sub = X_train_inner.loc[:, sub_inner]; 
                        X_test_inner_sub = X_test_inner.loc[:, sub_inner]; 
                    
                        # At this point we can standardise 
                        if algo in ['LR_reg', 'LR_no_reg', 'SVM', 'NB']:
                            X_train_inner_sub, X_test_inner_sub = standardise(X_train_inner_sub, X_test_inner_sub)
                    elif method in ['AE', 'PCA']:
                        #reduced_features = get_reduced_features(cv_outer, X, y, method)
                        X_train_inner_sub = reduced_features[fold_index][1]; X_test_inner_sub = reduced_features[fold_index][2]
                        X_train = reduced_features[fold_index][3]; X_test = reduced_features[fold_index][4]

                    space = return_parameter_space(algo, cardinality)
                    trials = Trials()
                    
                    best = fmin(fn=obj_fns[algo + '_' + cardinality], space=space,
                        algo=partial(mix.suggest, 
                                     p_suggest=[(.1, rand.suggest), (.2, anneal.suggest), (.7, tpe.suggest),]),
                                     max_evals=100, trials=trials, rstate=hyperopt_rstate)

                    # Retrieve the best parameters
                    best_params = space_eval(space, best)
                    if algo in ['LR_reg', 'LR_no_reg']:
                        if method not in ["AE", "PCA"]:
                            X_train, X_test = standardise(X_train.loc[:, sub], X_test.loc[:, sub])
                        best_model = LogisticRegression(random_state=42, n_jobs=-1, **best_params) # tree_method='hist', 
                    elif algo == 'SVM':
                        if method not in ["AE", "PCA"]:
                            X_train, X_test = standardise(X_train.loc[:, sub], X_test.loc[:, sub])
                        best_model = SVC(random_state=42, probability=True, **best_params)
                    elif algo == 'RF':
                        if method not in ["AE", "PCA"]:
                            X_train, X_test = np.array(X_train.loc[:, sub]), np.array(X_test.loc[:, sub])
                        best_model = RandomForestClassifier(random_state=42, **best_params) # tree_method='hist'
                    elif algo == 'NB':
                        if method not in ["AE", "PCA"]:
                            X_train, X_test = standardise(X_train.loc[:, sub], X_test.loc[:, sub])
                        best_model = GaussianNB(**best_params)
                    elif algo == 'XGB':
                        if method not in ["AE", "PCA"]:
                            X_train, X_test = np.array(X_train.loc[:, sub]), np.array(X_test.loc[:, sub])
                        best_model = XGBClassifier(random_state=42, **best_params)
                             
                    best_model.fit(X_train, y_train)
                    
                    # evaluate model on the hold out dataset
                    y_pred = best_model.predict(X_test)
                    # Get predicted probabilities
                    if cardinality == 'binary':
                        y_probas = best_model.predict_proba(X_test)[::, 1]
                    elif cardinality == 'multinomial':
                    	y_probas = best_model.predict_proba(X_test)
                    outer_predictions[method][subset]['Fold predictions'].append((y_pred))
                    outer_predictions[method][subset]['Fold probabilities'].append((y_probas))
                    outer_predictions[method][subset]['Fold test'].append((y_test))
                # Print feature subset progress    
                fpr, tpr, thresholds = roc_curve(np.concatenate(outer_predictions[method][subset]['Fold test']), 
                                                 np.concatenate(outer_predictions[method][subset]['Fold probabilities']))
                auc_score = auc(fpr, tpr)
                print("Features: {}, AUC: {} - Algo: {}, FS: {}".format(subset, np.round(auc_score, 4), algo, method))
                       
            # Summarize the estimated performance of the model over nested CV outer test sets
            results_dicts = {method: {key: None for key in subsets}}
            for subset in subsets:
                results_dicts[method][subset] = get_and_record_scores(outer_predictions[method][subset], cardinality)
            if not os.path.exists(results_path):            
                os.makedirs(results_path)
            
            save_results_dictionary(results_dicts, results_path + 'results_feature_ranking_' + str(algo) + '_' + str(method) + '.pkl')        
            print("Duration for {}: {}".format(str(algo), time.time() - start))
            
            display_results_table_and_graph(results_dicts, algo, method, results_path)
