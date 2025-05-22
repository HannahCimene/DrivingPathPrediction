# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:51:25 2025

@author: Hannah Cimene
"""


import pandas as pd

url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(url)

### Printing the original data
print("\n***************\nOriginal DataFrame Info:\n")
df.info()
print("\n***************\nOriginal DataFrame:\n")
print(df)

## Printing the transformed data
## Removing the data we don't need
# removing data where speed = 0
transformed_df = df[df['speed'] != 0].copy()
print("\n***************\nFinal Transformed DataFrame Info:\n")
transformed_df.info()
print("\n***************\nFinal Transformed DataFrame without speed 0:\n")
print(transformed_df)

# removing the duplicate data
transformed_df = transformed_df.drop_duplicates(subset=['ts', 'lat', 'lon'])
print("\n***************\nFinal Transformed DataFrame Info:\n")
transformed_df.info()
print("\n***************\nFinal Transformed DataFrame without speed 0:\n")
print(transformed_df)

# change the Dtype, this is needed for training the model
transformed_df['ts'] = pd.to_datetime(transformed_df['ts'])
print("\n***************\nFinal Transformed DataFrame Info:\n")
transformed_df.info()

# drop the duplicates
transformed_df = transformed_df.drop_duplicates(subset=['ts', 'lat', 'lon'])
print("\n***************\nFinal Transformed DataFrame Info:\n")
transformed_df.info()

vids_to_remove = ['X102480610', 'X103580610', 'X1027116', 'X103880610', 'X1057113']
transformed_df = transformed_df[~transformed_df['vid'].isin(vids_to_remove)]
print("\n***************\nFinal Transformed DataFrame Info:\n")
transformed_df.info()

# Find the datapoint(s) you want to delete
points_to_remove = transformed_df[(transformed_df['vid'] == 'X108370610') & (transformed_df['lon'] > 100.3)]
# Remove the points
transformed_df = transformed_df.drop(points_to_remove.index)
print("\n***************\nFinal Transformed DataFrame Info:\n")
transformed_df.info()


# print("\nFinal Transformed DataFrame Head:")
# print(transformed_df.head())

# df = df[['vid', 'ts', 'lat', 'lon']].copy()
# df = df.sort_values(by='ts')
# df = df.reset_index(drop=True)

# # Remove specified vids
# vids_to_remove = ['X102480610', 'X103580610', 'X1027116', 'X103880610', 'X1057113']
# df = df[~df['vid'].isin(vids_to_remove)]
# df = df.reset_index(drop=True)