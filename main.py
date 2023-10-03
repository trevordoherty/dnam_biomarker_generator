
import argparse
from compare_ML_models import *
from get_inputs import *
from pdb import set_trace


def main():
    """Main method."""
    parser = argparse.ArgumentParser(description="Script accepts a filepath string and a list of strings (ML algos) as arguments.")
    parser.add_argument("Input_File_Path", nargs='?', default="check_string_for_empty",
                         help="A string argument denoting the filepath of the input data.")
    parser.add_argument("Results_File_Path", nargs='?', default="check_string_for_empty",
                         help="A string argument denoting the filepath of the results data.")
    parser.add_argument('ML_List', type=str, help="A comma-separated list of strings denoting the ML algos you wish to compare.")
    args = parser.parse_args()
    
    algos = args.ML_List.split(',')

    # Read in input data file which contains both the CpGs and "Label" columns
    input_data = read_data(args.Input_File_Path) 

    # Simulate having 4 classes (Labels) to check multinomial capability of algos
    #input_data['Label'] = np.random.randint(0, 4, input_data.shape[0])
    
    best_algo = choose_ML_algorithm(input_data, args.Results_File_Path, algos)
    



if __name__ == "__main__":
    main()
