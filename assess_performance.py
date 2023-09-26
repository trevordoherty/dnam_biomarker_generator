

"""
Desc: This script calculates metrics and performance scores.
      
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve


def get_and_record_scores(outer_predictions, results):
    """Generate a range of relevant performance metrics."""
    all_test = np.concatenate(outer_predictions['Fold test'])
    all_pred = np.concatenate(outer_predictions['Fold predictions'])
    all_probas = np.concatenate(outer_predictions['Fold probabilities'])
    acc = accuracy_score(all_test, all_pred)
    fpr, tpr, thresholds = roc_curve(all_test, all_probas)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(all_test, all_pred)
    sens = cm[1][1] / (cm[1][1] + cm[1][0])
    spec = cm[0][0] / (cm[0][0] + cm[0][1])
    prec = cm[1][1] / (cm[1][1] + cm[0][1])
    print(f'Sens: {sens:.3f}, Spec: {spec:.3f}, Prec: {prec:.3f}, Acc: {acc:.3f}, AUC: {auc_score:.3f}')
    results['auc'].append(auc_score); results['sens'].append(sens); results['spec'].append(spec);
    results['prec'].append(prec); results['acc'].append(acc); results['All test'].append(all_test)
    results['All pred'].append(all_pred); results['All probas'].append(all_probas)
    # plt.plot(fpr, tpr); plt.show()
    return results


def save_results_dictionary(outer_results, filepath):
    """Save dictionary of results """
    with open(filepath, 'wb') as f: pickle.dump(outer_results, f)
    #with open(filepath, 'rb') as f:
    #    loaded_dict = pickle.load(f)


def load_results_dictionary(filepath):
    """Load dictionary of results """
    with open(filepath, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


