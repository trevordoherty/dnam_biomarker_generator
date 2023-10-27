
import argparse
from feature_reduction_nested_cv import *
from compare_ML_models import *
from get_inputs import *
from pdb import set_trace


def main():
    """Main method."""
    parser = argparse.ArgumentParser(description="Script accepts a filepath string and a list of strings (ML algos) as arguments.")
    parser.add_argument("Input_File_Path", nargs='+', default="check_string_for_empty",
                         help="A string argument denoting the filepath of the input data file/s.")
    parser.add_argument("Results_File_Path", default="check_string_for_empty",
                         help="A string argument denoting the filepath of the results data.")
    parser.add_argument('ML_List', type=str, help="A comma-separated list of strings denoting the ML algos you wish to compare.")
    parser.add_argument('Reduction', type=str, help='A string indicating whether to optimise models using a range of feature reduction' + 
                        'methods - pass "Yes" or "No". If "Yes", then a single ML algorithms should be passed. If "No". multiple ML algorithms could be passed.')
    args = parser.parse_args()
    
    algos = args.ML_List.split(',')
    if args.Reduction == "Yes":
        reducer = True
    else:
        reducer=False

    # Read in input data file which contains both the CpGs and "Label" columns
    input_data = read_data(args.Input_File_Path)
    
    # Simulate having 4 classes (Labels) to check multinomial capability of algos
    #input_data['Label'] = np.random.randint(0, 4, input_data.shape[0])
    if reducer == False:
        if len(args.Input_File_Path) == 1: 
            best_algo_nestedcv = assess_ML_algorithm_nested_cv(input_data, args.Results_File_Path, algos, reducer)
        elif len(args.Input_File_Path) > 1:
            best_algo_nestedcv = assess_ML_algorithm_nested_cv(input_data, args.Results_File_Path, algos, reducer)
            best_algo_test = assess_ML_algorithm_indep_test(input_data, args.Results_File_Path, algos)
    elif reducer == True:
        if len(args.Input_File_Path) == 1: 
            assess_feature_selection_nested_cv(input_data, args.Results_File_Path, algos)
        #elif len(args.Input_File_Path) > 1:
            """
            compare_feature_reduction_methods(input_data, args.Results_File_Path, algos, reducer)
            call feature reduction method comparison - returns the best model i.e., in the form of results of the comparative analysis
            Train the best FS + ML model on the full training data, test on indep data - returns the results, maybe the model i.e., the estimator/classifier
            The user of the package will want to have a prediction model which they can save down - what about missing values CpGs - how to handle those?
            Like DNAmTL - do we just indciate that the available CpGs be used e.g., 118 of 140 were used by us when testing EXTEND ...
            """


if __name__ == "__main__":
    main()
