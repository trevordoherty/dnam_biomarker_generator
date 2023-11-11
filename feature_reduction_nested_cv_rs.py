

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
from feature_reduction_methods_rs import *
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
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import time
from topfeatureselector import *
from xgboost import XGBClassifier
import warnings
hyperopt_rstate = np.random.RandomState(42)
warnings.filterwarnings("ignore")

def assess_feature_selection_nested_cv_gs(input_data, results_path, ml_algo_list):
    """Returns the best ML algroithm.

    Evaluated using a nested CV analysis on the training data set.
    The algorithm that achieves the best ROC AUC score is chosen.
    """
    feat_max = 10000
    best_algo = {'LR_no_reg': [], 'LR_reg': [], 'SVM': [], 'RF': [], 'XGB': [], 'NB': []}

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
    
    methods = ["MI"] # ,  "Variance", "MI", "ANOVA", "Random Forest" "PCA", "AE"
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
                for fold_index, (train_ix, test_ix) in enumerate(cv_outer.split(X)):
                    # split data
                    X_train, X_test = X.iloc[:, 0:feat_max].iloc[train_ix, :], X.iloc[:, 0:feat_max].iloc[test_ix, :]  # X.iloc[:, 0:500]
                    y_train, y_test = y[train_ix], y[test_ix]
                    
                    
                    pipes = {'LR_reg': Pipeline([('scaler', StandardScaler()),
                        ('feature_selector', TopNFeatureSelector(subset, reduced_features[fold_index])), ('classifier', LogisticRegression(random_state=42, n_jobs=-1))]),
                         'LR_no_reg': Pipeline([('scaler', StandardScaler()),('classifier', LogisticRegression(random_state=42, n_jobs=-1))]),
                         'SVM': Pipeline([('scaler', StandardScaler()),('classifier', SVC(random_state=42, probability=True))]),
                         'NB': Pipeline([('scaler', StandardScaler()),('classifier', GaussianNB())]),
                         'RF': Pipeline([('classifier', RandomForestClassifier(random_state=42))]),
                         'XGB': Pipeline([('classifier', XGBClassifier(random_state=42))])
                         }
                                       

                    # Configure the inner cross-validation
                    cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
                    parameters = return_parameter_space_gs(algo, cardinality)
                    
                    model = RandomizedSearchCV(pipes[algo], parameters, cv=cv_inner, n_iter=50, random_state=42).fit(X_train, y_train)
                    set_trace()

                    if method in ["Random Forest", "Variance", "MI", "ANOVA"]:
                        # At this point, get the feature ranking on X_train_inner and X_train
                        fold_importances = reduced_features[fold_index][1].index[0:subset]
                        sub = reduced_features[fold_index][2].index[0:subset]
                    
                        X_train_inner_sub = X_train_inner.loc[:, sub_inner]; 
                        X_test_inner_sub = X_test_inner.loc[:, sub_inner]; 
                    
                        # At this point we can standardise 
                        model = RandomizedSearchCV(pipes[algo], parameters, cv=cv_inner, n_iter=50).fit(X_train, y_train)


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
            
            save_results_dictionary(results_dicts, results_path + 'results_feature_ranking_' + str(algo) + '_' + str(method) + '_rs.pkl')        
            print("Duration for {}: {}".format(str(algo), time.time() - start))
            
            display_results_table_and_graph(results_dicts, algo, method, results_path)
