
"""
This script compares a range of feature reduction methodologies in conjunction with an ML algorithm in 
a nested CV analysis.

Author: Trevor Doherty
Date: 24/10/23


"""

from feature_reduction_nested_cv import *
from compare_ML_models import *
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, SelectFdr, VarianceThreshold


'''
def get_feature_rankings(X_train, X_test, y_train, method, feature_cols):
    """Return feature rankings."""
    if method == "MI":
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        # Learn the relationship from training data
        fs.fit(X_train, y_train)
        # Get the features ranked by MI score
        feature_scores = dict(zip(feature_cols, fs.scores_))
        feature_scores_sorted = pd.Series(feature_scores).sort_values(ascending=False)
        return feature_scores_sorted#
''' 


def standardise(X_train_inner, X_test_inner):
    """Apply standardisation within the inner train/test split."""
    # Scaling
    sc = StandardScaler()
    X_train_inner = sc.fit_transform(X_train_inner)
    X_test_inner = sc.transform(X_test_inner)
    return X_train_inner, X_test_inner


def get_feature_rankings(cv_outer, X, y, algo, method):
    """Get feature rankings in advance and map into CV loops."""
    ranked_features = []
    for fold_index, (train_ix, test_ix) in enumerate(cv_outer.split(X)):
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]  # X.iloc[:, 0:500]
        y_train, y_test = y[train_ix], y[test_ix]
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = \
            train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    
        feature_cols = X_train_inner.columns
        # At this point we can standardise 
        if method == "MI":
	        fs = SelectKBest(score_func=mutual_info_classif, k='all')
	        # Learn the relationship from training data
	        fs.fit(X_train_inner, y_train_inner) 
	        # Get the features ranked by MI score
	        feature_scores = dict(zip(feature_cols, fs.scores_))
	        feature_scores_sorted_sub = pd.Series(feature_scores).sort_values(ascending=False)#
	        
	        fs = SelectKBest(score_func=mutual_info_classif, k='all')
	        # Learn the relationship from training data
	        fs.fit(X_train, y_train)
	        # Get the features ranked by MI score
	        feature_scores = dict(zip(feature_cols, fs.scores_))
	        feature_scores_sorted = pd.Series(feature_scores).sort_values(ascending=False)
	        ranked_features.append((fold_index, feature_scores_sorted_sub, feature_scores_sorted))
        elif method == 'ANOVA':
            fs = SelectFdr(f_classif, alpha=0.01)
            fs.fit(X_train_inner, y_train_inner)
            # Get the features ranked by MI score
            feature_scores = dict(zip(feature_cols, fs.scores_))
            feature_scores_sorted_sub = pd.Series(feature_scores).sort_values(ascending=False)

            fs = SelectFdr(f_classif, alpha=0.01)
            fs.fit(X_train, y_train) 
            # Get the features ranked by MI score
            feature_scores = dict(zip(feature_cols, fs.scores_))
            feature_scores_sorted = pd.Series(feature_scores).sort_values(ascending=False)
            ranked_features.append((fold_index, feature_scores_sorted_sub, feature_scores_sorted))
        elif method == "Variance":
            fs = VarianceThreshold()
            fs.fit(X_train_inner)
            feature_scores = dict(zip(feature_cols, fs.variances_))
            feature_scores_sorted_sub = pd.Series(feature_scores).sort_values(ascending=False)

            fs = VarianceThreshold()
            fs.fit(X_train)
            feature_scores = dict(zip(feature_cols, fs.variances_))
            feature_scores_sorted = pd.Series(feature_scores).sort_values(ascending=False)
            ranked_features.append((fold_index, feature_scores_sorted_sub, feature_scores_sorted))
    return ranked_features
	                