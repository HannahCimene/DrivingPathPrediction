import pandas as pd

###############################################################################

# CSV = 'PROBE-202410/20241001.csv.out'
# PROBE-202410/20241001.csv.out
# C:/school_2024-2025/internship/Task3/PROBE-202410/20241001.csv.out
CSV = "C:\\School_2024-2025\\Internship\\Task3\\PROBE-202410\\20241001.csv.out"

df = pd.read_csv(CSV)
print(df.head())

df.columns = ['vid', 'valgps', 'lat', 'lon', 'ts', 'speed', 'dir', 'hirelight', 'engineactive']

df.info()

df.nunique()

mv = df.isnull().sum()
print(mv)

###############################################################################

duplicates = df.duplicated()
duplicate_rows = df[duplicates]
print(duplicate_rows)

df_no_duplicates_last = df.drop_duplicates(keep='last')
df = df_no_duplicates_last
df

###############################################################################

df = df[(df['speed'] > 0) & (df['speed'] <= 125)]
print('The vehicle is driving at a decent speed')
print(df.head())
df.info()

###############################################################################

df = df[df['valgps'] == 1]
print('The valid GPS points')
print(df.head())
df.info()

###############################################################################

lat_counts_sorted_by_latitude  = df['lat'].value_counts().sort_index()
print(lat_counts_sorted_by_latitude)

lon_counts_sorted_by_latitude  = df['lon'].value_counts().sort_index()
print(lon_counts_sorted_by_latitude)

###############################################################################

df = df[df['hirelight'] == 0]
print('There are possibly passengers in the vehicle')
print(df.head())
df.info()

###############################################################################

# 1. Count Data Points per Vehicle
vehicle_counts = df['vid'].value_counts()
vehicles_to_remove = vehicle_counts[vehicle_counts < 15].index

# 3. Filter the Original DataFrame to keep only vehicles with 15 or more data points
df_filtered = df[~df['vid'].isin(vehicles_to_remove)]

df_filtered_counts = df_filtered['vid'].value_counts()
print(df_filtered_counts)

# Now, df_filtered contains only the data for vehicles with 15 or more data points.

# You can check the number of unique vehicles before and after filtering:
print(f"Number of unique vehicles before filtering: {df['vid'].nunique()}")
print(f"Number of unique vehicles after filtering: {df_filtered['vid'].nunique()}")

# You can also check the minimum count of data points per vehicle in the filtered DataFrame:
vehicle_counts_filtered = df_filtered['vid'].value_counts()
print(f"Minimum data points per vehicle after filtering: {vehicle_counts_filtered.min()}")

df_filtered.info()

df_filtered.head()

df = df_filtered

###############################################################################

# Group by vehicle_id and sort by time_stamp
df = df.groupby('vid').apply(lambda x: x.sort_values(by='ts'))
df = df.reset_index(drop=True)
df.info()
df.nunique()


print(df.head())






