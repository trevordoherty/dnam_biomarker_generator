

import argparse
from assess_performance import *
import os
from pdb import set_trace


def main():
    """Main method."""
    parser = argparse.ArgumentParser(description="Script accepts a filepath string and a list of strings (ML algos) as arguments.")
    parser.add_argument("Results_File_Path", nargs='+', default="check_string_for_empty",
                         help="A string argument denoting the filepath of the results folder.")

    path = "C:/Users/User/Desktop/D Drive/dnam_pypi/results/"
    
    tasks = ["dnam_only/short", "dnam_only/long", "dnam_only/disease",
             "dnam_age_sex_bcc/short", "dnam_age_sex_bcc/long", "dnam_age_sex_bcc/disease"]
    
    dicts = []    
    results = []
    for task in tasks:
        for files in os.walk(path + task):
            for f in files[2]:
                if f.startswith('results2'):
                    tmp = load_results_dictionary(path + task + "/" + f)
                    dicts.append((task, f, tmp))
                    # Find the index of the first underscore and the first dot
                    first_underscore_index = f.index("_")
                    first_dot_index = f.index(".")

                    # Extract the substring between the first underscore and the first dot
                    extracted_string = f[first_underscore_index + 1 : first_dot_index]
                    results.append([task, extracted_string, tmp['auc'], tmp['sens'], tmp['spec'],
                    	            tmp['acc'], tmp['prec']])
    
    results_df = pd.DataFrame(results, columns=['Task', 'ML Algoirthm', "AUC", "Sens", "Spec", "Acc", "Prec"])
    
    # Variance-based ranking for short recurrence (dnam_only)            
    for task in tasks:
    	for files in os.walk(path + task):
            for f in files[2]:
                if f.startswith('results_feature_ranking'):
                    tmp = load_results_dictionary(path + task + "/" + f)
                    
                    aucs = []
                    lines = []
                    for subset in range(100, 10001, 100):
                    	key = list(tmp.keys())[0]
                    	aucs.append(tmp[key][subset]['auc'])
                    print("Max AUC for {}: {} - {} features".format(task + key, max(aucs),
                                                                   (aucs.index(max(aucs)) + 1)*100))

                    line, = plt.plot(range(100, 10001, 100), aucs, label=key)
                    lines.append(line)
                
            plt.legend(loc='lower right')
            plt.ylim(0, 0.8)
            plt.xlabel('#Features'), plt.ylabel('AUC')
            plt.title("AUC vs. #Features - " + task)
            plt.show()

            
    
    """                
    
    for subset in range(100, 10001, 100):
        aucs.append(tmp['Variance'][subset]['auc'])
    set_trace()    
    """
    

    # Pull out FS results for short also - set FS in motion for long and disease
    # LR looks good in these cases ...

if __name__ == "__main__":
    main()
    