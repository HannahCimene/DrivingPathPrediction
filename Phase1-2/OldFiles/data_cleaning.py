# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:38:31 2025

@author: Hannah Cimene
"""

import csv
import pandas as pd

from matplotlib import pyplot

from sklearn.preprocessing import StandardScaler, MinMaxScaler

url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'

df = pd.read_csv(url)
df.head()  # Print the first few rows to verify

# Save the DataFrame to a local CSV file
df.to_csv('gps-data.csv', index=False) # index=False prevents writing the dataframe index as a column.

df.info()

df.nunique()

duplicates = df.duplicated()
duplicate_rows = df[duplicates]
print(duplicate_rows)

df_no_duplicates_last = df.drop_duplicates(keep='last')

df = df_no_duplicates_last
df

df = df.rename(columns={'vid': 'vehicle_id', 'ts': 'time_stamp', 'lat': 'latitude', 'lon': 'longitude'})

print(df)

# Assuming your DataFrame is named 'df' and the timestamp column is 'timestamp'
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

import pandas as pd

# Assuming 'df' is your DataFrame and 'speed' is the column name
df = df[df['speed'] != 0]

# Now 'df' contains only rows where 'speed' is not equal to 0
print(df.head())

#remove the speed
df = df.drop('speed', axis=1)

df.info()

# Group by vehicle_id and sort by time_stamp
df = df.groupby('vehicle_id').apply(lambda x: x.sort_values(by='time_stamp'))

# Reset index to remove multi-index created by groupby.
df = df.reset_index(drop=True)

df.info()

df.to_csv('cleaned_data.csv')