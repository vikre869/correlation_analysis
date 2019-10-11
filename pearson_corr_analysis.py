import os
import csv
import time
import numpy as np
import pandas as pd
from scipy import stats
from functools import partial
from multiprocessing import Pool 
from help_functions import memory_optimize,memory_optimize2

def filter_pearson(x,y):
    return stats.pearsonr(x,y)[0]

def calculate_pearson(max_div, n):

    static_file = 'AMD_features_binary.csv'
    dynamic_file = 'extracted_features_all.csv'

    headers = []
    with open(dynamic_file, "rt") as f:
        reader = csv.reader(f)
        headers = next(reader)

    # Remove sample and variety as columns
    headers = headers[2:]
    max_headers = len(headers)/max_div

    if n == 1:
        columns = headers[:int(max_headers*n)]
    else:
        columns = headers[int(max_headers*(n-1)):int(max_headers*n)]

    df_chunk = pd.read_csv(dynamic_file, engine='c', low_memory=False, chunksize=5000, usecols=['sample', 'variety'] + columns)
    chunk_list = []
    for chunk in df_chunk:
        # Run memory optimization
        opt_chunk = memory_optimize2(chunk)
        # Append optimized chunk to chunk_list
        chunk_list.append(opt_chunk)

    # Concatinate the chunks into one dataframes
    df_dynamic = pd.concat(chunk_list)

    # load entire static dataset
    df_static = pd.read_csv(static_file, engine='c', low_memory=False)
    # Drop irrelevant data from static dataset
    df_static.drop(df_static.columns[1:4], axis=1, inplace=True)
    # Optimize since all data are binary
    df_static = memory_optimize(df_static)

    # Get dynamic and static columns
    dynamic_features = list(df_dynamic.columns)
    static_features = list(df_static.columns)

    # Merge static and dynamic datasets
    df_static_part = pd.DataFrame(index=df_dynamic.index, columns=static_features, dtype=np.uint16)
    for index, row in df_static_part.iterrows():
        df_static_part.loc[index] = df_static.loc[index[1]]
    df_concat = pd.concat([df_dynamic, df_static_part], axis=1)

    # Clear RAM from unnecessary datasets
    del [df_static, df_dynamic]

    # Leave 1 core free from calculation
    max_workers = os.cpu_count() - 1
    pool = Pool(processes = max_workers)

    # result dataframe
    df_correlations = pd.DataFrame(index=dynamic_features, columns=static_features)

    # Get each dynamic feature values in to lists
    dynamic_feature_values = []
    for column in dynamic_features:
        dynamic_feature_values.append(list(df_concat[column])) 

    print(f'INFO: Starting calculations.')
    start = time.time()
    # Run pearson correlation
    for static_feature in static_features:
        static_feature_values = list(df_concat[static_feature])
        results = pool.map(partial(filter_pearson, y=static_feature_values), dynamic_feature_values)
        df_correlations.loc[dynamic_features, static_feature] = results
    end = time.time()
    print(f'INFO: Calculations were completed in: {(end - start)/60} minutes.')
    # Write results to csv
    df_correlations.to_csv(f'Feature_Correlations{n}_{max_div}.csv', sep=',', encoding='utf-8')
