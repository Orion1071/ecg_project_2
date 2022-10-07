from timeit import default_timer as timer
from datetime import timedelta


import pandas as pd
import numpy as np
import wfdb
import ast
import pickle
import matplotlib.pyplot as plt 
import sys





def pickler(obj_save, filepath:str) -> None:
    """
    This function use pickle to save objects
    This can be used to save a python object to reuse later
    This function does not do any error handeling and will overwrite if the file exists

    Parameter
    object: any object that is to be saved as binary/byte strean
    filepath: an address or filepath to save the pickle file

    Return 
    None
    """
    print("pickling")
    pickle_file_fh = open(filepath, "wb")
    # for x in obj_save:
    #     print(id(x))
    pickle.dump(obj=obj_save, file=pickle_file_fh,protocol=0)
    pickle_file_fh.close()
    print("pickle done")

def unpickler(filepath: str):
    """
    This function is used to open the pickle file
    This function does not do any error handeling

    Parameter
    filepath: a filepath to the pickle file

    Returns
    A python object from pickle file
    """
    print("pickle opening")
    return pickle.load(open(filepath, "rb"))

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
