
"""
This script compares a range of feature reduction methodologies in conjunction with an ML algorithm in 
a nested CV analysis.

Author: Trevor Doherty
Date: 24/10/23


"""
from autoencoder import *
from compare_ML_models_hp import *
from compare_ML_models_rs import *
from feature_reduction_nested_cv_hp import *
from feature_reduction_nested_cv_rs import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, SelectFdr, VarianceThreshold
from sklearn.preprocessing import StandardScaler


def get_reduced_features(cv_outer, X, y, method):
    """Get feature rankings in advance and map into CV loops."""
    reduced_features = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for fold_index, (train_ix, test_ix) in enumerate(cv_outer.split(X)):
        X_train, X_test = X.iloc[:, 0:500].iloc[train_ix, :], X.iloc[:, 0:500].iloc[test_ix, :]  # X.iloc[:, 0:500]
        y_train, y_test = y[train_ix], y[test_ix]
        
        cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        for fold_idx, (train_ix, test_ix) in enumerate(cv_inner.split(X_train, y_train)):
            X_train_fold, y_train_fold = X_train.iloc[train_ix], y_train.iloc[train_ix]
            X_test_fold, y_test_fold = X_train.iloc[test_ix], y_train.iloc[test_ix]
                
            feature_cols = X_train_fold.columns
            # At this point we can standardise 
            if method == "MI":
                """
                fs = SelectKBest(score_func=mutual_info_classif, k='all')
                # Learn the relationship from training data
                fs.fit(X_train_fold, y_train_fold) 
                # Get the features ranked by MI score       
                feature_scores = dict(zip(feature_cols, fs.scores_))
                feature_scores_fold = pd.Series(feature_scores).sort_values(ascending=False)#
                """
                fs = SelectKBest(score_func=mutual_info_classif, k='all')
                # Learn the relationship from training data
                fs.fit(X_train, y_train)
                # Get the features ranked by MI score
                feature_scores = dict(zip(feature_cols, fs.scores_))
                feature_scores_train = pd.Series(feature_scores).sort_values(ascending=False)
                #reduced_features[fold_index][fold_idx] = [fold_index, fold_idx, feature_scores_fold, feature_scores_train]
                reduced_features[fold_index] = [fold_index, feature_scores_train]
            elif method == 'ANOVA':
                """
                fs = SelectFdr(f_classif, alpha=0.01)
                fs.fit(X_train_fold, y_train_fold)
                # Get the features ranked by MI score
                feature_scores = dict(zip(feature_cols, fs.scores_))
                feature_scores_fold = pd.Series(feature_scores).sort_values(ascending=False)
                """
                fs = SelectFdr(f_classif, alpha=0.01)
                fs.fit(X_train, y_train) 
                # Get the features ranked by MI score
                feature_scores = dict(zip(feature_cols, fs.scores_))
                feature_scores_train = pd.Series(feature_scores).sort_values(ascending=False)
                #reduced_features.append((fold_index, fold_idx, feature_scores_fold, feature_scores_train))
                reduced_features.append((fold_index, feature_scores_train))
            elif method == "Variance":
                """
                fs = VarianceThreshold()
                fs.fit(X_train_fold)
                feature_scores = dict(zip(feature_cols, fs.variances_))
                feature_scores_fold = pd.Series(feature_scores).sort_values(ascending=False)
                """
                fs = VarianceThreshold()
                fs.fit(X_train)
                feature_scores = dict(zip(feature_cols, fs.variances_))
                feature_scores_train = pd.Series(feature_scores).sort_values(ascending=False)
                #reduced_features.append((fold_index, fold_idx, feature_scores_fold, feature_scores_train))
                reduced_features.append((fold_index, feature_scores_train))
            elif method == "Random Forest":
                parameter_grid = {'max_samples': [0.5, 0.75, 0.99],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'n_estimators': [10, 50, 100, 200],
                                  'max_depth': [5, 10, 20]
                                  }
                """
                model = RandomForestClassifier(random_state=0)
                search = RandomizedSearchCV(model, parameter_grid,
                                            scoring='neg_mean_squared_error',
                                            refit=True, n_jobs=-1, verbose=2, random_state=0)
                model = search.fit(X_train_fold, y_train_fold) 
                feature_scores_fold = \
                    pd.Series(dict(zip(X.iloc[:, :-1].columns.values,
                                       model.best_estimator_.feature_importances_)),
                                       name='Score').sort_values(ascending=False)
                """
                model = RandomForestClassifier(random_state=0)
                search = RandomizedSearchCV(model, parameter_grid,
                                            scoring='neg_mean_squared_error',
                                            refit=True, n_jobs=-1, verbose=2, random_state=0)
                model = search.fit(X_train, y_train) 
                feature_scores_train = \
                    pd.Series(dict(zip(X.iloc[:, :-1].columns.values,
                                       model.best_estimator_.feature_importances_)),
                                       name='Score').sort_values(ascending=False)
                #reduced_features.append((fold_index, fold_idx, feature_scores_fold, feature_scores_train))
                reduced_features.append((fold_index, feature_scores_train))
            elif method == "PCA":
                """
                # Apply to X_train_fold, X_test_fold
                pca = PCA(n_components=0.95)
                X_train_fold = pca.fit_transform(X_train_fold)
                X_test_fold = pca.transform(X_test_fold)
                """
                # Apply to X_train, X_test
                pca = PCA(n_components=0.95)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                reduced_features.append((fold_index, fold_idx, X_train_fold, X_test_fold, X_train, X_test))
                reduced_features.append((fold_index, X_train, X_test))
            elif method == "AE":
                X_train_fold, X_test_fold = \
                    return_latent_spaces(X_train_fold, X_test_fold, y_train_fold, y_test_fold)  
                X_train, X_test = \
                    return_latent_spaces(X_train, X_test, y_train, y_test)  
                reduced_features.append((fold_index, fold_idx, X_train_fold, X_test_fold, X_train, X_test))
                
    return reduced_features
	                