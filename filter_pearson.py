import pandas as pd
import numpy as np
import csv
import json
from help_functions import memory_optimize3


def filter(filename, n_varieties):    
    df_chunk = pd.read_csv(filename, engine='c', low_memory=False, chunksize=5000)
    chunk_list = []
    for chunk in df_chunk:
        # Run memory optimization
        opt_chunk = memory_optimize3(chunk)
        # Append optimized chunk to chunk_list
        chunk_list.append(opt_chunk)
    df_concat = pd.concat(chunk_list)

    # Drop all static features that is present in less than n varieties
    with open('/home/viktor/correlation_analysis/static_feature_count.json') as json_file:
        data = json.load(json_file)
    for key in data.keys():
        if data[key] < n_varieties:
            df_concat.drop(key, axis=1, inplace=True)
    columns = df_concat.columns
    list_of_tuples = []

    for index, row in df_concat.iterrows():
        numpy_array = row.to_numpy(dtype=np.float32)
        temp = []
        for i in range(len(columns)):
            temp.append((f'{index}:{columns[i]}', numpy_array[i]))
        list_of_tuples += temp
    return list_of_tuples

def main():
    all_correlations = []
    for i in range(1,11):
        correlations = filter(f'Feature_Correlations{i}_10.csv', 1)
        all_correlations += correlations
    all_correlations.sort(reverse=True , key=lambda tup: tup[1])
    print('\\begin{landscape}')
    print('\\begin{table}[h]')
    print('\\centering')
    print('\\caption{Dynamic Static Correlation} \label{tab:corr_load}')
    print('\\begin{tabular}{|p{14cm}|c|c|}')
    print('\\hline')
    print('\\multicolumn{3}{|c|}{\\textbf{load}} \\\\ \\hline')			
    print('\\textbf{Dynamic Features} & \\textbf{Static Features} & \\textbf{Correlation}	\\\\ \hline')
    i = 1
    for pair in all_correlations[:25]:
        dynamic_static = pair[0].split(':')
        dynamic_static[0] = dynamic_static[0].replace('_','\\_')
        dynamic_static[1] = dynamic_static[1].replace('_','\\_')
        print(f'{dynamic_static[0]} & {dynamic_static[1]} & {"{:12.2f}".format(pair[1])} \\\\ \\hline')
        i += 1
    print('\\end{tabular}')
    print('\\end{table}')
    print('\\end{landscape}')
#with open('sorted_correlations.csv', 'w', newline='') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(all_correlations)


if __name__ == "__main__":
    main()