import pandas as pd
import numpy as np
from sys import getsizeof
import json

def print_size(df):
    start_size = getsizeof(df)/(1024.0**3)
    print('Dataframe size: %2.2f GB'%start_size)

def memory_optimize(df):
    df = df.set_index(['Variety'])
    df = df.fillna(0)
    df = df.astype(np.int16)
    return df

def memory_optimize2(df):
    df = df.set_index(['sample', 'variety'])
    df = df.fillna(0)
    df = df.astype(np.int16)
    return df

def memory_optimize3(df):
    df = df.set_index(df.columns[0])
    df = df.fillna(0)
    df = df.astype(np.float32)
    return df

def feature_extraction(n):
    data = {}
    with open('/home/viktor/mahalanobis/final_calculations/feature_variances.json') as json_file:
        data = json.load(json_file)
        json_file.close()
    print(len(data))

    list_of_tuples = []
    # Add items as tuples in a list
    for key in data:
        list_of_tuples.append((key, data[key]))
    # sort list based on variance
    list_of_tuples.sort(reverse=True ,key=lambda tup: tup[1])  # sorts in place
    # Select the N first in the list (the ones with the highest variance)
    list_of_tuples = list_of_tuples[:n]
    # Make a list of the remaining features in list_of_tuples without their variance.
    for i in range(len(list_of_tuples)):
        list_of_tuples[i] = list_of_tuples[i][0]
    return list_of_tuples

