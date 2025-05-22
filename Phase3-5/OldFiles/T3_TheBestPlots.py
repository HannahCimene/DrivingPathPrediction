# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:22:48 2025

@author: Hannah Cimene
"""

#pip install tensorflow_addons

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input, Attention
from sklearn.metrics import mean_squared_error, mean_absolute_error


###########################################

# the dataframe is preprocessed
# we remove the speed where it is 0 because the taxi is parked
# we change the timestamp to value datetime
# we remove duplicates of the ts, lat and lon because this is redundant
# we sort the timestamps by time
def transform_taxi_data(df):
    """Transforms taxi data."""
    df = df[df['speed'] != 0].copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.drop_duplicates(subset=['ts', 'lat', 'lon'])
    df = df[['vid', 'ts', 'lat', 'lon']].copy()
    df = df.sort_values(by='ts')
    df = df.reset_index(drop=True)
    return df

# we create the sequence
# the lenght is 10, so there will be 10 x points and 10 y points that will
#be given before predicting the next point
def create_sequences(df, sequence_length):
    """Creates input sequences from DataFrame."""
    sequences = []
    for vid, group in df.groupby('vid'):
        lats = group['lat'].tolist()
        lons = group['lon'].tolist()
        for i in range(len(lats) - sequence_length):
            seq_lats = lats[i:i + sequence_length]
            seq_lons = lons[i:i + sequence_length]
            target_lat = lats[i + sequence_length]
            target_lon = lons[i + sequence_length]
            sequences.append((seq_lats, seq_lons, target_lat, target_lon))
    return sequences

# we need the data to be prepared before the training of the model
# we first store the sequences
# then convert the X and y to a NumPy array (correct shape for RNN/LSTM)
def prepare_data(sequences, sequence_length):
    """Prepares data for RNN/LSTM model."""
    X_lat, X_lon, y_lat, y_lon = [], [], [], []
    for seq in sequences:
        X_lat.append(seq[0])
        X_lon.append(seq[1])
        y_lat.append(seq[2])
        y_lon.append(seq[3])
    X = np.stack((X_lat, X_lon), axis=-1)
    y = np.stack((y_lat, y_lon), axis=-1)
    return X, y

# to build a RNN model
# change the dense layers, activation function, optimizer or loss
def build_simple_rnn_model(input_shape):
    """Builds SimpleRNN model."""
    model = Sequential()
    model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

# to build a LSTM model
# change the dense layers, activation function, optimizer or loss
def build_lstm_model(input_shape):
    """Builds LSTM model."""
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    return model

###### still add dropout like this one: model.add(LSTM(100, activation='relu', return_sequences=True, dropout=0.2))

# to build a LSTM model with attention
def build_lstm_model_with_attention(input_shape):
    """Builds LSTM model with attention using tf.keras.layers.Attention."""
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    # Apply Attention layer
    attention_output = Attention()([lstm_out, lstm_out]) # query and value are both lstm_out
    # Add a Dense layer after the attention layer
    attention_output_dense = Dense(50, activation='relu')(attention_output)
    # Concatenate the LSTM output and attention output
    combined_output = tf.keras.layers.Concatenate(axis=-1)([lstm_out, attention_output_dense])
    # Add more Dense layers
    dense1 = Dense(100, activation='relu')(combined_output)
    dense2 = Dense(50, activation='relu')(dense1)
    # Apply a dense layer to the combined output
    outputs = Dense(2)(dense2)
    # Take the last time step output.
    outputs = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(clipvalue=0.5, learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# evaluate the model
# by using the test data to calculate the Mean Absolute Error and the
#Root Mean Square Error
# the actual and predicted values get plotted
def evaluate_model(model, X_test, y_test, model_name, epochs, batch_size, sequence_length):
    """Evaluates the model and plots results, including date/time, epochs, and batch size."""
    y_pred = model.predict(X_test)
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    print(f"{model_name} MAE: {mae}")
    print(f"{model_name} RMSE: {rmse}")
    # Plotting (example: first 100 test points)
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_np[:100, 0], y_test_np[:100, 1], label='Actual', marker='o')
    plt.plot(y_pred_np[:100, 0], y_pred_np[:100, 1], label='Predicted', marker='x')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Get current date and time
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    plt.title(f'{model_name} - Actual vs. Predicted Trajectories\nDate/Time: {date_time_str}\nEpochs: {epochs}, Batch Size: {batch_size}, Sequence Length: {sequence_length}')
    plt.show() #show each plot individually.

############################

# Main execution
url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(url)
df.head()
transformed_df = transform_taxi_data(df)
transformed_df.head()
transformed_df.info()

vids_to_remove = ['X102480610', 'X103580610', 'X1027116', 'X103880610', 'X1057113']
transformed_df = transformed_df[~transformed_df['vid'].isin(vids_to_remove)]
transformed_df = transformed_df.reset_index(drop=True)
transformed_df.info()

sequence_length=20

sequences = create_sequences(transformed_df, sequence_length)

train_size = int(len(sequences) * 0.8)
train, test = sequences[:train_size], sequences[train_size:]

X_train, y_train = prepare_data(train, sequence_length)
X_test, y_test = prepare_data(test, sequence_length)

############################

# variables for training the model
sequence_length = 20
input_shape = (sequence_length, 2)
epochs = 50
batch_size = 64

# SimpleRNN Model
print("\nSimpleRNN Model:")
simple_rnn_model = build_simple_rnn_model(input_shape)
simple_rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
evaluate_model(simple_rnn_model, X_test, y_test, 'SimpleRNN', epochs, batch_size, sequence_length)

# LSTM Model
print("\nLSTM Model:")
lstm_model = build_lstm_model(input_shape)
lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
evaluate_model(lstm_model, X_test, y_test, 'LSTM', epochs, batch_size, sequence_length)

# LSTM Model with Attention
print("\nLSTM Model with Attention:")
lstm_attention_model = build_lstm_model_with_attention(input_shape)
lstm_attention_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
evaluate_model(lstm_attention_model, X_test, y_test, 'LSTM with Attention', epochs, batch_size, sequence_length)