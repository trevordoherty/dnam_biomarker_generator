
import numpy as np
from pdb import set_trace
from sklearn.base import BaseEstimator, TransformerMixin


class TopNFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=100, precomputed_importances_list=None):
        self.top_n = top_n
        self.precomputed_importances_list = precomputed_importances_list
        self.selected_features_ = None

    def fit(self, X, y=None):
        # Use the precomputed importances for the specific fold
        ranked_features = self.precomputed_importances_list[1].index[0:self.top_n]
        self.selected_features_ = ranked_features
        # Select the top N features
        #top_n_indices = np.argsort(feature_importances)[-self.top_n:]
        #self.selected_features_ = np.zeros(X.shape[1], dtype=bool)
        #self.selected_features_[top_n_indices] = True
        
        return self

    def transform(self, X):
        set_trace()
        return X[:, self.selected_features_]


# C:/Users/User/venvs/ibd/Scripts/activate
#"D:/ibd_cluster_data/ibd_pypi_dnam_only_short.pkl" "C:/Users/User/Desktop/D Drive/dnam_pypi/src" "LR_reg" "Yes" "GS"