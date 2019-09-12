import os
import numpy as np
import pandas as pd
from scipy import stats
from functools import partial
from multiprocessing import Pool  

# Pearson correlation computations without p-value
def pearson_corr(x,y):
    return stats.pearsonr(x,y)[0]

static_file = 'AMD_features_binary.csv'
dynamic_file = 'dataset200.csv'
# load entire static dataset
df_static = pd.read_csv(static_file, engine='c', low_memory=False)
df_static.set_index(['Variety'], inplace=True)

# Drop irrelevant data from static dataset
df_static.drop(df_static.columns[:3], axis=1, inplace=True)

# load part of dynamic dataset
df_dynamic = pd.read_csv(dynamic_file, engine='c', low_memory=False)
df_dynamic.set_index(['sample', 'variety'], inplace=True)

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

# Run pearson correlation
for static_feature in static_features:
    static_feature_values = df_concat[static_feature]
    results = pool.map(partial(stats.pearsonr, x=static_feature_values), dynamic_feature_values)
    df_correlations.loc[dynamic_features, static_feature] = results

# Write results to csv
df_concat.to_csv('Feature_Correlations.csv', sep=',', encoding='utf-8')
