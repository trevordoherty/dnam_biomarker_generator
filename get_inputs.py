

"""
Desc: This script reads in the specified DNAm data and labels for the condition of interest.
The location of the file is passed as an argument when running the main.py script.

"""

import pandas as pd
from pdb import set_trace

def read_data(path):
    input_data_list = []
    if len(path) == 1:
        input_data = pd.read_pickle(path[0])
        input_data_list.append(input_data)
    elif len(path) == 2:
        train_data = pd.read_pickle(path[0])
        test_data = pd.read_pickle(path[1])
        input_data_list.append(train_data); input_data_list.append(test_data)
    return input_data_list
