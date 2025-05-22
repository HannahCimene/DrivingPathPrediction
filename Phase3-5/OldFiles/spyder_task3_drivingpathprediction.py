#pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error # Added mean_absolute_errorimport math
import matplotlib.pyplot as plt
import math

import pandas as pd
import numpy as np

def transform_taxi_data(df):
    """
    Transforms taxi data by combining lat/lon, removing speed 0, converting ts to datetime, and handling duplicates.

    Args:
      df: Pandas DataFrame with taxi data (vid, ts, lat, lon, speed).

    Returns:
      Pandas DataFrame with transformed data (vid, ts, lat_lon).  # Added vid to return
    """

    # 1. Remove rows where speed is 0
    df = df[df['speed'] != 0].copy()

    # 2. Convert 'ts' to datetime
    df['ts'] = pd.to_datetime(df['ts'])

    # 3. Combine 'lat' and 'lon' into a single tuple column
    df['lat_lon'] = list(zip(df['lat'], df['lon']))

    # 4. Remove duplicate rows based on 'ts' and 'lat_lon'
    df = df.drop_duplicates(subset=['ts', 'lat_lon'])

    # 5. Select only 'vid', 'ts' and 'lat_lon' columns
    df = df[['vid', 'ts', 'lat_lon']].copy() #added vid

    #6. Sort by time
    df = df.sort_values(by = 'ts')

    #7. reset index.
    df = df.reset_index(drop = True)

    return df

def get_train_test_from_df(df, train_percent=0.8):
    """
    Splits time series data from a DataFrame into train and test sets.

    Args:
        df: Pandas DataFrame containing the time series data.
        train_percent: Percentage of data for training.

    Returns:
        train_data, test_data, data (original scaled data)
    """

    # Extract the 'lat_lon' column for scaling
    data = np.array(df['lat_lon'].to_list())

    # Reshape if necessary for MinMaxScaler
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    n = len(data_scaled)

    # Calculate split point
    train_split = int(n * train_percent)

    # Split the data
    train_data = data_scaled[:train_split]
    test_data = data_scaled[train_split:]

    return train_data, test_data, data_scaled

def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[range(time_steps * rows_x)]
    X = np.reshape(X, (rows_x, time_steps, data.shape[1]))  # Adjusted for lat_lon tuples
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def print_error(trainY, testY, train_predict, test_predict):
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))

def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Lat/Lon scaled') #Changed label.
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(url)
transformed_df = transform_taxi_data(df)

### REMOVING THE VID'S WITH ALMOST NO POINTS
# Assuming 'transformed_df' is your DataFrame
vids_to_remove = ['X102480610', 'X103580610', 'X1027116', 'X103880610', 'X1057113']
# Filter out rows with the specified vids
transformed_df = transformed_df[~transformed_df['vid'].isin(vids_to_remove)]
# Optional: Reset the index after removal
transformed_df = transformed_df.reset_index(drop=True)

time_steps = 3

# Group by vehicle_id
grouped_data = transformed_df.groupby('vid')

# Iterate through each vehicle group
for vehicle_id, vehicle_group in grouped_data:
    print(f"\nTraining model for Vehicle ID: {vehicle_id}")

    # Remove 'vid' column only for training and testing data preparation
    vehicle_group_no_vid = vehicle_group.drop('vid', axis=1)

    train_data, test_data, data = get_train_test_from_df(vehicle_group_no_vid)
    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)

    # Create and train model
    model = create_RNN(hidden_units=3, dense_units=2, input_shape=(time_steps, data.shape[1]),
                       activation=['tanh', 'tanh'])
    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

    # Make predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    # Print error (RMSE)
    print_error(trainY, testY, train_predict, test_predict)

    # Calculate and print MAE
    train_mae = mean_absolute_error(trainY, train_predict)
    test_mae = mean_absolute_error(testY, test_predict)
    print('Train MAE: %.3f MAE' % (train_mae))
    print('Test MAE: %.3f MAE' % (test_mae))

    plot_result(trainY, testY, train_predict, test_predict)
    plt.title(f"Vehicle ID: {vehicle_id} - Actual vs. Predicted")
    plt.show()