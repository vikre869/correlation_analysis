import json
import pandas as pd
import numpy as np
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
columns = df_static.columns
families = []
for index,row in df_static.iterrows():
    families.append(index.split('/')[0])
df_static['Family'] = families

df_families = pd.DataFrame(index=set(families), columns=columns)
for family in families:
    summa = df_static.loc[df_static['Family'] == family].sum()
    df_families.loc[family] = summa

df_families = df_families.astype(np.bool_)
df_families = df_families.astype(np.int)
summa = df_families.sum(axis=0)


print(df_families.head())
summa.to_json('static_feature_family_count.json')