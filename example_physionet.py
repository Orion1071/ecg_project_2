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




def main():
    start_main = timer()
    np.set_printoptions(threshold=sys.maxsize)
    path = '/scratch/thurasx/ptb_xl/'
    sampling_rate=101
    filtered_filepath = '/scratch/thurasx/ptb_xl/filtered_ecg_id.csv'
    # load and convert annotation data
    oY = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    fY = pd.read_csv(filtered_filepath, index_col='ecg_id')
    Y = oY.merge(fY, on='ecg_id')
    #convert str to dictonary
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    #save pcikel object
    pickle_file_path = '/scratch/thurasx/ptb_xl/all_waves_data.pcl'


    # Load raw signal data
    # X = load_raw_data(Y, sampling_rate, path)


    #save pickle file
    # pickler(obj_save=X, filepath=pickle_file_path)
    #open used pickle file
    X = unpickler(pickle_file_path)



    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv')
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass\
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


    # # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]    
    print(X_train)    
    print("X_train_done")

    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    print("y_train_done")
    
    # pickler(obj_save=X_train, filepath=f"{path}X_train.pcl")
    # pickler(obj_save=y_train, filepath=f"{path}y_train.pcl")
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    print("X_test_done")
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    print("y_test_done")
    # pickler(obj_save=X_test, filepath=f"{path}X_test.pcl")
    # pickler(obj_save=y_test, filepath=f"{path}y_test.pcl")
    end_main = timer()
    print(f'main done in: {timedelta(seconds=end_main - start_main)}\n')


if __name__== "__main__" :
    main()



"""

    print(Y['diagnostic_superclass'])
    print(Y.strat_fold)
    test plotting and see the results
    tmp = unpickler("/scratch/thurasx/x1.pcl")
    tmp = tmp[:]
    plt.figure(figsize=(100,10))
    plt.plot(tmp)
    plt.savefig("test.png")

    # psrint(type(X_train[0]))
    # print(X_train[0].shape)
    # print(X_train[0].ndim)

"""