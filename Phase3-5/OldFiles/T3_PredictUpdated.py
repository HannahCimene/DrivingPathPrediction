import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------

# the dataframe is preprocessed
# we remove the speed where it is 0 because the taxi is parked
# we change the timestamp to value datetime
# we remove duplicates of the ts, lat and lon because this is redundant
# we sort the timestamps by time
def transform_taxi_data(df):
    """Transforms taxi data and removes specified vids."""
    df = df[df['speed'] != 0].copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.drop_duplicates(subset=['ts', 'lat', 'lon'])
    df = df[['vid', 'ts', 'lat', 'lon']].copy()
    df = df.sort_values(by='ts')
    # Remove specified vids
    vids_to_remove = ['X102480610', 'X103580610', 'X1027116', 'X103880610', 'X1057113']
    df = df[~df['vid'].isin(vids_to_remove)]
    # Find the datapoint(s) you want to delete
    points_to_remove = df[(df['vid'] == 'X108370610') & (df['lon'] > 100.3)]
    # Remove the points
    df = df.drop(points_to_remove.index)
    
    df = df.reset_index(drop=True)

    return df

#------------------------------------------------------------------------------


CSV_PATH = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(CSV_PATH)
# df

# dfv = df[(df['vid']=="X0997116")&(df['speed']>0)]
# dfv.info()
# dfv

dfv = transform_taxi_data(df)
print("\nFinal Transformed DataFrame Info:")
dfv.info()
dfv

## for plotting the values
# dfv.sort_values('ts')[['lat']].plot()

# def createXY(dataset,n_past):
#     dataX = []
#     dataY = []
#     for i in range(n_past, len(dataset)):
#             dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
#             dataY.append(dataset[i,:])
#     return np.array(dataX),np.array(dataY)

def createXY(df, n_past, lat_col='lat', lon_col='lon'):
    """
    Creates input sequences and target values for time series prediction,
    taking into account different vehicles (grouping by 'vid').

    Args:
        df (pd.DataFrame): The input DataFrame with 'vid', timestamp, latitude, longitude, etc.
        n_past (int): The number of past time steps to use for prediction.
        lat_col (str): The name of the latitude column.
        lon_col (str): The name of the longitude column.

    Returns:
        tuple: A tuple containing two dictionaries:
            - grouped_dataX (dict): A dictionary where keys are 'vid' and values are
              NumPy arrays of input sequences for that vehicle.
            - grouped_dataY (dict): A dictionary where keys are 'vid' and values are
              NumPy arrays of target values (next [lat, lon]) for that vehicle.
    """
    grouped_dataX = {}
    grouped_dataY = {}

    for vid, group in df.groupby('vid'):
        # Sort the group by timestamp to maintain chronological order
        group = group.sort_values(by='ts')

        # Extract latitude and longitude values as a NumPy array
        relevant_data = group[[lat_col, lon_col]].values

        dataX_vid = []
        dataY_vid = []

        if len(relevant_data) > n_past:
            for i in range(n_past, len(relevant_data)):
                dataX_vid.append(relevant_data[i - n_past:i, :])  # Use both lat and lon as features
                dataY_vid.append(relevant_data[i, :])          # Target is the next [lat, lon]

            grouped_dataX[vid] = np.array(dataX_vid)
            grouped_dataY[vid] = np.array(dataY_vid)
        else:
            print(f"Warning: Vehicle '{vid}' has fewer than {n_past + 1} data points and will be skipped.")

    return grouped_dataX, grouped_dataY

## for plotting the values, here because else it won't plot when running it all
# dfv.sort_values('ts')[['lon']].plot()

scaler_lat = StandardScaler()
scaler_lon = StandardScaler()

slat = scaler_lat.fit_transform(dfv[['lat']])
slon = scaler_lon.fit_transform(dfv[['lon']])

dfvs = np.column_stack((slat, slon))

dfvs[:10]

X, Y = createXY(dfvs, 10)

X

ylat = Y[:,0]
ylon = Y[:,1]

print(ylat[:10])
print(ylon[:10])

X.shape
Y.shape

#####################################################

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Stop after 10 epochs with no improvement
    restore_best_weights=True,  # Restore model weights from the epoch with the best validation loss
    verbose=1 #added verbose to print when early stopping occurs.
)

#####################################################

####### Only first model optimized, need to do the rest

### Predict the lat with LSTM model
epochs=30
batch_size=1
shape=(10, 2)

model = Sequential([
    Input(shape=shape),
    LSTM(10),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylat[1:], epochs=epochs, batch_size=batch_size, verbose=2)

ylat_pred_scaled = model.predict(X)
ylat_pred = scaler_lat.inverse_transform(ylat_pred_scaled)

plt.plot(dfv['lat'].values[10:], label='real') # Plot original scale for 'real'
plt.plot(ylat_pred, label='predict')
plt.title(f'Predicted lat with LSTM (StandardScaler)\nEpochs={epochs}, Batch size={batch_size}')
plt.xlabel('Time Steps')
plt.ylabel('Latitude')
plt.legend()
plt.show()

#####################################################

### Predict the lon with LSTM model

model = Sequential()
model.add(LSTM(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylon[1:], epochs=20, batch_size=1, verbose=2)

ylon_pred = model.predict(X)

plt.plot(ylon_pred[:,0], label='predict')
plt.plot(ylon, label='real')
plt.title(label='Predicted lon with LSTM')
plt.show()

# scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0]

#####################################################

### Predict the lat with RNN model

model = Sequential()
model.add(SimpleRNN(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylat[1:], epochs=20, batch_size=1, verbose=2)

ylat_pred = model.predict(X)

plt.plot(scaler_lat.inverse_transform(np.reshape(ylat_pred[:,1], (1, len(ylat))))[0], label='predict')
plt.plot(scaler_lat.inverse_transform(np.reshape(ylat, (1, len(ylat))))[0], label='real')
plt.title(label='Predicted lat with RNN')
plt.show()

scaler_lat.inverse_transform(np.reshape(ylat, (1, len(ylat))))[0]

#####################################################

### Predict the lon with RNN model

model = Sequential()
model.add(SimpleRNN(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylon[1:], epochs=20, batch_size=1, verbose=2)

ylon_pred = model.predict(X)

plt.plot(scaler_lon.inverse_transform(np.reshape(ylon_pred[:,1], (1, len(ylon))))[0], label='predict')
plt.plot(scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0], label='real')
plt.title(label='Predicted lon with RNN')
plt.show()

scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0]

#####################################################

# Trying the attention mechanism in LSTM

### Predict the lat with LSTM with attention

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Concatenate, Lambda
import tensorflow.keras.optimizers
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def build_lstm_with_attention_sequential_like(input_shape):
    """Builds an LSTM model with attention, mimicking a Sequential structure using the Functional API."""
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(inputs)

    # Apply Attention layer
    attention_output = Attention()([lstm_out, lstm_out])  # query and value are both lstm_out

    # Add a Dense layer after the attention layer
    attention_output_dense = Dense(50, activation='relu')(attention_output)

    # Concatenate the LSTM output and attention output
    combined_output = Concatenate(axis=-1)([lstm_out, attention_output_dense])

    # Add more Dense layers
    dense1 = Dense(100, activation='relu')(combined_output)
    dense2 = Dense(50, activation='relu')(dense1)

    # Apply a dense layer to the combined output
    outputs = Dense(2)(dense2)

    # Take the last time step output.
    outputs = Lambda(lambda x: x[:, -1, :])(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = tensorflow.keras.optimizers.Adam(clipvalue=0.5, learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Sample data (replace with your actual data)
X = np.random.rand(100, 10, 2)
ylon = np.random.rand(100)  # Assuming ylon is a 1D array

# Scale the target variable (important for neural networks)
scaler_lon = MinMaxScaler()
ylon_reshaped = ylon.reshape(-1, 1)  # Reshape ylon
ylon_scaled = scaler_lon.fit_transform(ylon_reshaped)

# Build and train the LSTM model with attention
input_shape = (X.shape[1], X.shape[2])
model = build_lstm_with_attention_sequential_like(input_shape)
model.fit(X[:-1], ylon_scaled[1:], epochs=20, batch_size=1, verbose=2)

# Make predictions
ylon_pred_scaled = model.predict(X)
ylon_pred = scaler_lon.inverse_transform(ylon_pred_scaled)

# Plotting (adjust if your prediction target is different)
plt.plot(ylon_pred, label='predict')
plt.plot(scaler_lon.inverse_transform(ylon_scaled), label='real')
plt.title(label='Predicted lon with LSTM with Attention (Sequential-like)')
plt.legend()
plt.show()

print(scaler_lon.inverse_transform(ylon_scaled))


