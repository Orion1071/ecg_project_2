from timeit import default_timer as timer
from datetime import timedelta


import pandas as pd
import numpy as np
import wfdb
import ast
import pickle
import matplotlib.pyplot as plt 
import sys
import ecg_library_v1 as v1


def main():
    start_main = timer()
    
    pickle_file_path = '/scratch/thurasx/ptb_xl/all_waves_data.pcl'
    X = v1.unpickler(pickle_file_path)
    end_main = timer()
    print(f'main done in: {timedelta(seconds=end_main - start_main)}\n')

if __name__== "__main__" :
    main()


"""

Read scp codes and convert to list

Y = pd.read_csv("/scratch/thurasx/ptb_xl/filtered_scp.csv")
print(type(Y.scp_codes[0]))
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
keys = []
for i in range(len(Y)):
    keys.append(list(Y.scp_codes[i].keys())[0])
reference_scp_file_path = "/scratch/thurasx/ptb_xl/reference_scp_list.pcl"
v1.pickler(obj_save=keys, filepath=reference_scp_file_path)

"""