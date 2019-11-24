import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import log10
from help_functions import memory_optimize3


def filter(filename):
    df_chunk = pd.read_csv(filename, engine='c', low_memory=False, chunksize=5000)
    chunk_list = []
    for chunk in df_chunk:
        # Run memory optimization
        opt_chunk = memory_optimize3(chunk)
        # Append optimized chunk to chunk_list
        chunk_list.append(opt_chunk)
    df_concat = pd.concat(chunk_list)
    list_of_corrs = []
    columns = df_concat.columns

    for index,row in df_concat.iterrows():
        numpy_array = row.to_numpy(dtype=np.float32)
        for i in range(len(numpy_array)):
            list_of_corrs.append((f'{index} & {columns[i]}', numpy_array[i]))

    return list_of_corrs

def main():
    all_correlations = []
    for i in range(1,11):
        correlations = filter(f'Feature_Correlations{i}_10.csv')
        all_correlations += correlations
    all_correlations.sort(reverse=True ,key=lambda tup: tup[1])  # sorts in place

    for elem in all_correlations[:69]:
        print('\hline')
        print(f'{elem[0]} \\\\')

if __name__ == "__main__":
    main()