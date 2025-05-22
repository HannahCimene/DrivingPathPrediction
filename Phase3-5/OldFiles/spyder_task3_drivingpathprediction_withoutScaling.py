# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:12:10 2025

@author: Hannah Cimene
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def transform_taxi_data(df):
    """Transforms taxi data."""
    df = df[df['speed'] != 0].copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df['lat_lon'] = list(zip(df['lat'], df['lon']))
    df = df.drop_duplicates(subset=['ts', 'lat_lon'])
    df = df[['vid', 'ts', 'lat_lon']].copy()
    df = df.sort_values(by='ts')
    df = df.reset_index(drop=True)
    return df

def create_sequences(df, sequence_length=10):
    """Creates input sequences from DataFrame."""
    sequences = []
    for vid, group in df.groupby('vid'):
        points = group['lat_lon'].tolist()
        for i in range(len(points) - sequence_length):
            sequences.append(points[i:i + sequence_length + 1])  # Include target
    return sequences

def prepare_data(sequences, sequence_length):
    """Prepares data for RNN/LSTM model."""
    X, y = [], []
    for seq in sequences:
        X.append(seq[:-1])
        y.append(seq[-1])
    return np.array(X), np.array(y)

def build_simple_rnn_model(input_shape):
    """Builds SimpleRNN model."""
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_model(input_shape):
    """Builds LSTM model."""
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and plots results."""
    y_pred = model.predict(X_test)
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)

    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Plotting (example: first 100 test points)
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_np[:100, 0], y_test_np[:100, 1], label='Actual', marker='o')
    plt.plot(y_pred_np[:100, 0], y_pred_np[:100, 1], label='Predicted', marker='x')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Actual vs. Predicted Trajectories')
    plt.show()

# Main execution
url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(url)
transformed_df = transform_taxi_data(df)

vids_to_remove = ['X102480610', 'X103580610', 'X1027116', 'X103880610', 'X1057113']
transformed_df = transformed_df[~transformed_df['vid'].isin(vids_to_remove)]
transformed_df = transformed_df.reset_index(drop=True)

sequences = create_sequences(transformed_df, sequence_length=10)

train_size = int(len(sequences) * 0.8)
train, test = sequences[:train_size], sequences[train_size:]

X_train, y_train = prepare_data(train, 10)
X_test, y_test = prepare_data(test, 10)

# SimpleRNN Model
print("\nSimpleRNN Model:")
simple_rnn_model = build_simple_rnn_model(input_shape=(10, 2))
simple_rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
evaluate_model(simple_rnn_model, X_test, y_test)

# LSTM Model
print("\nLSTM Model:")
lstm_model = build_lstm_model(input_shape=(10, 2))
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
evaluate_model(lstm_model, X_test, y_test)