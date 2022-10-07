from timeit import default_timer as timer
from datetime import timedelta

import pandas as pd
import csv
import json


def read_csv(filepath : str):
    return pd.read_csv(filepath)


# def label_category(label_in):
    

def label_checker(ECG_labels : list, str_in : str) -> bool :
    for label in ECG_labels:
        if label in str_in:
            return label
    return None

def data_filter(data_in, ECG_labels : list):
    
    ret_ecg_id = []
    data_cat = {}
    # pandas version
    id_scp = data_in.loc[:,["ecg_id","scp_codes"]]
    i = 1
    for data in id_scp["scp_codes"]:
        label = label_checker(ECG_labels=ECG_labels, str_in=data)
        if label != None:
            ret_ecg_id.append(i)
            if label in data_cat:
                data_cat[label] += 1
            else:
                data_cat[label] = 1
        i += 1

    # print(len(data_in))
    # print(len(ret_ecg_id))
    # print(data_cat)
    return ret_ecg_id, data_cat

def main():
    start_main = timer()
    ECG_labels_interested = [	"NORM"  ,"IMI"  ,"ASMI" ,"ISC"  ,"ISCAL",
                    "ILMI"  ,"AMI"  ,"ALMI" ,"ISCIN","INJAS",
                    "LMI"   ,"ISCIL","ISCAS","INJAL","ISCLA",
                    "IPLMI" ,"ISCAN","IPMI" ,"INJIN","INJLA",
                    "PMI"   ,"INJIL","STD_" ,"ISC_" ,"NST_",
                    'QWAVE' ,"NT_"  ,'INVT' ,"STE_"]

    labels_touse = ['NDT'    , 'NST_', 'DIG' , 'LNGQT', 'NORM', 'IMI', 
                        'ASMI'  , 'LVH' , 'LAFB', 'ISC_', 'IRBBB', '1AVB',
                        'IVCD', 'ISCAL', 'CRBBB', 'CLBBB', 'ILMI', 'LAO/LAE', 
                        'AMI', 'ALMI', 'ISCIN', 'INJAS', 'LMI', 'ISCIL', 
                        'LPFB', 'ISCAS', 'INJAL', 'ISCLA', 'RVH', 'ANEUR', 
                        'RAO/RAE', 'EL', 'WPW', 'ILBBB', 'IPLMI', 'ISCAN', 
                        'IPMI', 'SEHYP', 'INJIN', 'INJLA', 'PMI', '3AVB', 
                        'INJIL', '2AVB']

    ECG_labels = set(ECG_labels_interested).intersection(labels_touse)
    print("inter : ", len(ECG_labels_interested))
    print("touse :", len(labels_touse))
    print("ecg :", len(ECG_labels))
    print(ECG_labels)

    ptbxl_csv_filepath = '/scratch/thurasx/ptb_xl/ptbxl_database.csv'
    data = read_csv(ptbxl_csv_filepath)
    
    ret_ecg_id, data_cat = data_filter(data_in= data, ECG_labels=ECG_labels)
    filterd_data = data.loc[data["ecg_id"].isin(ret_ecg_id)]
    print(len(filterd_data))
    print(filterd_data)
    filtered_ecg_file_path = '/scratch/thurasx/ptb_xl/filtered_ecg_id.csv'
    filtered_scp_file_path = '/scratch/thurasx/ptb_xl/filtered_scp.csv'
    filterd_data.to_csv(filtered_ecg_file_path,index=False, columns=['ecg_id']) 
    print("ecg_file saved to ", filtered_ecg_file_path)
    filterd_data.to_csv(filtered_scp_file_path,index=False, columns=['scp_codes']) 
    print("scp_file saved to ", filtered_scp_file_path)
    end_main = timer()
    print(f'main done in: {timedelta(seconds=end_main - start_main)}\n')


if __name__== "__main__" :
    main()