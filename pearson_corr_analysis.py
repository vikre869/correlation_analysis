import pandas as pd
import os
from multiprocessing import Pool  
from scipy import stats
from functools import partial

def pearson_corr(x,y):
    return stats.pearsonr(x,y)[0]

static_file = 'AMD_features_binary.csv'
dynamic_file = 'dynamic_features.csv'
# load entire static dataset
df_static = pd.read_csv(static_file, engine='c', low_memory=False)
df_static.set_index(['Family', 'Variety'], inplace=True)
df_static.drop(df_static.columns[:3], inplace=True)

# load part of dynamic dataset
df_dynamic = pd.read_csv(dynamic_file, engine='c', low_memory=False)
df_dynamic.set_index(['sample','variety'])

# fill pool with work of feature combinations
max_workers = os.cpu_count() - 1
pool = Pool(processes = max_workers)

# result dataframe
dynamic_features = list(df_dynamic.columns)
static_features = list(df_static.columns)
df_correlations = pd.DataFrame(index=dynamic_features, columns=static_features)

# Get each dynamic feature values in to lists
dynamic_feature_values = []
for column in dynamic_features:
    dynamic_feature_values.append(list(df_dynamic[column])) 

# Create work data
for static_feature in static_features:
    static_feature_values = df_static[static_feature]
    # same static, different dynamic feature
    results = pool.map(partial(stats.pearsonr, x=static_feature_values), dynamic_feature_values)
    df_correlations.loc[dynamic_features, static_feature] = results


