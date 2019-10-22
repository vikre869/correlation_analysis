import json
import pandas as pd
from help_functions import memory_optimize
from list_varieties import successful_varieties

static_file = f'AMD_features_binary.csv'

# load entire static dataset
df_static = pd.read_csv(static_file, engine='c', low_memory=False)
# Drop irrelevant data from static dataset
df_static.drop(df_static.columns[1:4], axis=1, inplace=True)
# Optimize since all data are binary
df_static = memory_optimize(df_static)

varieties_to_remove = set(df_static.index) - set(successful_varieties())
df_static.drop(list(varieties_to_remove), axis=0, inplace=True)
summa = df_static.sum(axis=0)

summa.to_json('static_feature_count.json')