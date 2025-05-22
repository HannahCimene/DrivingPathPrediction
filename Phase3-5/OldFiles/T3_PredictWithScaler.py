import numpy as np
import pandas as pd

# import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

#####################################################

CSV_PATH = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'

df = pd.read_csv(CSV_PATH)

df

dfv = df[(df['vid']=="X0997116")&(df['speed']>0)]

dfv.sort_values('ts')[['lat']].plot()

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,:])
    return np.array(dataX),np.array(dataY)

#here because else it won't plot when running it all
dfv.sort_values('ts')[['lon']].plot()

scaler_lat = MinMaxScaler(feature_range=(0, 1))
scaler_lon = MinMaxScaler(feature_range=(0, 1))

slat = scaler_lat.fit_transform(dfv[['lat']])
slon = scaler_lon.fit_transform(dfv[['lon']])


print('lat', slat[:10])
print('lon', slon[:10])

dfvs = np.append(slat,slon, axis=1)

dfvs[:10]

X, Y = createXY(dfvs, 10)

X

ylat = Y[:,0]
ylon = Y[:,1]

print(ylat[:10])
print(ylon[:10])

X.shape

#####################################################

### Predict the lat with LSTM model

model = Sequential()
model.add(LSTM(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylat[1:], epochs=20, batch_size=1, verbose=2)

ylat_pred = model.predict(X)

plt.plot(scaler_lat.inverse_transform(np.reshape(ylat_pred[:,1], (1, len(ylat))))[0], label='predict')
plt.plot(scaler_lat.inverse_transform(np.reshape(ylat, (1, len(ylat))))[0], label='real')
plt.title(label='Predicted lat with LSTM')
plt.show()

scaler_lat.inverse_transform(np.reshape(ylat, (1, len(ylat))))[0]

#####################################################

### Predict the lon with LSTM model

model = Sequential()
model.add(LSTM(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylon[1:], epochs=20, batch_size=1, verbose=2)

ylon_pred = model.predict(X)

plt.plot(scaler_lon.inverse_transform(np.reshape(ylon_pred[:,1], (1, len(ylon))))[0], label='predict')
plt.plot(scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0], label='real')
plt.title(label='Predicted lon with LSTM')
plt.show()

scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0]

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


