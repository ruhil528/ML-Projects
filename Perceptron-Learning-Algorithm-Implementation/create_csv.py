import csv 
import numpy as np 
import pandas as pd 

# Generate random pandas  dataframe
np.random.seed(1)
df = pd.DataFrame(np.random.randn(1000,3), columns=['feature_1', 'feature_2', 'label'])
print(df)

# Replace values in label column with 1 df['label'] > 0 or -1 if df['label'] < 0
df.loc[df['label']>0, 'label'] = 1
df.loc[df['label']<=0, 'label'] = -1
print(df)

# pandas dataframe save to csv
df.to_csv('input1.csv', index=False)
