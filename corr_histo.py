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

    for row in df_concat.itertuples():
        numpy_array = list(row)[1:]
        list_of_corrs += numpy_array
    return list_of_corrs

def main():
    all_correlations = []
    for i in range(1,11):
        correlations = filter(f'Feature_Correlations{i}_10.csv')
        all_correlations += correlations
    

    bins = 20
    plt.rcParams.update({'font.size': 17})
    plt.hist(all_correlations, range=(-1,1), bins=bins, rwidth=0.95)
    plt.xlabel('Pearson Cofficient r')
    plt.ylabel('Static and Dynamic feature permutations (logarithmic scale)')
    plt.xticks(np.arange(-1, 1.1, step=0.1))
    plt.title("One-to-One Correlations")
    plt.yscale('log', nonposy='clip')
    plt.show()

    #print(all_correlations[:10])
    #hist, bin_edges = np.histogram(all_correlations, range=(-1,1), bins=bins)
    #print(f'Hist: {hist}')
    #print(f'bin_edges: {bin_edges}')


if __name__ == "__main__":
    main()