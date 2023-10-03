

"""
Desc: This script calculates metrics and performance scores.
      
"""

#import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve, roc_auc_score


def get_and_record_scores(outer_predictions, cardinality):
    """Generate a range of relevant performance metrics."""
    results = {}
    all_test = np.concatenate(outer_predictions['Fold test'])
    all_pred = np.concatenate(outer_predictions['Fold predictions'])
    all_probas = np.concatenate(outer_predictions['Fold probabilities'])
    acc = accuracy_score(all_test, all_pred)

    if cardinality == 'binary':
        fpr, tpr, thresholds = roc_curve(all_test, all_probas)
        auc_score = auc(fpr, tpr)
        cm = confusion_matrix(all_test, all_pred)
        sens = cm[1][1] / (cm[1][1] + cm[1][0])
        spec = cm[0][0] / (cm[0][0] + cm[0][1])
        prec = cm[1][1] / (cm[1][1] + cm[0][1])
        #print(f'Sens: {sens:.3f}, Spec: {spec:.3f}, Prec: {prec:.3f}, Acc: {acc:.3f}, AUC: {auc_score:.3f}')
        print('Sens: {}, Spec: {}, Prec: {}, Acc: {}, AUC: {}'.format(sens, spec, prec, acc, auc_score))
        results['auc'] = auc_score; results['sens']  = sens; results['spec'] = spec;
        results['prec'] = prec; results['acc'] = acc
        # plt.plot(fpr, tpr); plt.show()
    elif cardinality == 'multinomial':
        auc_score = roc_auc_score(all_test, all_probas, multi_class='ovr', average='micro')
        cm = confusion_matrix(all_test, all_pred)
        num_classes = cm.shape[0]
        for idx in range(num_classes):
            tp = cm[idx, idx]
            total_class = np.sum(cm[idx, :])
            sensitivity  = tp / total_class if total_class > 0 else 0.0
            results['sens' + str(idx)] = sensitivity
        results['acc'] = acc; results['auc'] = auc_score
    results['All test'] = all_test; results['All pred'] = all_pred; results['All probas'] = all_probas
    print('Results: {}'.format(results))
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


