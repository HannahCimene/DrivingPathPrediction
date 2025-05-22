# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:59:57 2025

@author: Hannah Cimene
"""

import numpy as np
import pandas as pd

# import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

CSV_PATH = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'

df = pd.read_csv(CSV_PATH)

df

dfv = df[(df['vid']=="X0997116")&(df['speed']>0)]

dfv.sort_values('ts')[['lat']].plot()

# dfv.sort_values('ts')[['lon']].plot()

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,:])
    return np.array(dataX),np.array(dataY)

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

### Predict the lat 

model = Sequential()
model.add(LSTM(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylat[1:], epochs=20, batch_size=1, verbose=2)

ylat_pred = model.predict(X)

plt.plot(scaler_lat.inverse_transform(np.reshape(ylat_pred[:,1], (1, len(ylat))))[0], label='predict')
plt.plot(scaler_lat.inverse_transform(np.reshape(ylat, (1, len(ylat))))[0], label='real')
plt.title(label='Predicted lat')
plt.show()

scaler_lat.inverse_transform(np.reshape(ylat, (1, len(ylat))))[0]

#####################################################

### Predict the lon

model = Sequential()
model.add(LSTM(10, input_shape=(10,2)))
model.add(Dense(2, activation=keras.ops.relu))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X[:-1], ylon[1:], epochs=20, batch_size=1, verbose=2)

ylon_pred = model.predict(X)

plt.plot(scaler_lon.inverse_transform(np.reshape(ylon_pred[:,1], (1, len(ylon))))[0], label='predict')
plt.plot(scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0], label='real')
plt.title(label='Predicted lon')
plt.show()

scaler_lon.inverse_transform(np.reshape(ylon, (1, len(ylon))))[0]



