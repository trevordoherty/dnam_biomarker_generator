

"""
Desc: This script reads in the specified DNAm data and labels for the condition of interest.
The location of the file is passed as an argument when running the main.py script.

"""

import pandas as pd

def read_data(path):
    input_data = pd.read_pickle(path)
    return input_data
