# -*- coding: utf-8 -*-
"""
@author: Hannah Cimene
"""
### Check if you are using the right versions
# import sys
# print("Python Version:", sys.version)

# import tensorflow as tf
# print("TensorFlow Version:", tf.__version__)
# print("Keras Version:", tf.keras.__version__)

### This should be installed
# Python Version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]
# TensorFlow Version: 2.19.0
# Keras Version: 3.9.2

# pip install tensorflow_addons
# pip install --upgrade tensorflow keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
# import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input, Attention, Lambda
# from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

###############################################################################
################################## Functions ##################################
###############################################################################

### Cleans and keeps relevant taxi data (vid, ts, lon)

def transform_taxi_data(df):
    """Transforms taxi data and removes specified vids for longitude."""
    df.columns = ['vid', 'valgps', 'lat', 'lon', 'ts', 'speed', 'dir', 'hirelight', 'engineactive']
    df = df.drop_duplicates(subset=['ts', 'lat', 'lon'])
    df = df[(df['speed'] > 0) & (df['speed'] <= 125)].copy()
    df = df[df['valgps'] == 1]
    df = df[df['hirelight'] == 0]
    vehicle_counts = df['vid'].value_counts()
    vehicles_to_remove = vehicle_counts[vehicle_counts < 15].index
    df = df[~df['vid'].isin(vehicles_to_remove)]
    df['ts'] = pd.to_datetime(df['ts'])
    df = df[['vid', 'ts', 'lon', 'lat']].copy()  # Keep only 'lon'
    df = df.sort_values(by='ts')
    df = df.reset_index(drop=True)
    return df

###############################################################################

### Creates longitude sequences for each vehicle

def create_sequences_lon_only(df, sequence_length):
    sequences = []
    for vid, group in df.groupby('vid'):
        lons = group['lon'].tolist()
        for i in range(len(lons) - sequence_length):
            seq_lons = lons[i:i + sequence_length]
            # target_lon = lons[i + sequence_length]
            sequences.append(seq_lons)
    return sequences

### Creates latitude sequences for each vehicle

def create_sequences_lat_only(df, sequence_length):
    sequences = []
    for vid, group in df.groupby('vid'):
        lats = group['lat'].tolist()
        for i in range(len(lats) - sequence_length):
            seq_lats = lats[i:i + sequence_length]
            # target_lat = lats[i + sequence_length]
            sequences.append(seq_lats)
    return sequences

###############################################################################

### Reshapes longitude sequences into input (X) and target (y) arrays for the model

def prepare_data_lon_only(sequences, sequence_length):
    X = np.array(sequences)
    y = np.array([seq[-1] for seq in sequences])
    X = np.reshape(X, (X.shape[0], sequence_length, 1))
    y = np.reshape(y, (-1, 1))
    return X, y

### Reshapes latitude sequences into input (X) and target (y) arrays for the model

def prepare_data_lat_only(sequences, sequence_length):
    X = np.array(sequences)
    y = np.array([seq[-1] for seq in sequences])
    X = np.reshape(X, (X.shape[0], sequence_length, 1))
    y = np.reshape(y, (-1, 1))
    return X, y

###############################################################################

### Splits data into train, test, and validation sets by vehicle ID

def create_train_test_val_sets_grouped_lon_only(df, sequence_length):
    vids = df['vid'].unique()
    train_vids = vids[:int(len(vids) * 0.8)]
    test_vids = vids[int(len(vids) * 0.8):int(len(vids) * 0.9)]
    val_vids = vids[int(len(vids) * 0.9):]

    train_sequences = []
    test_sequences = []
    val_sequences = []

    for vid, group in df.groupby('vid'):
        group = group.sort_values(by='ts')
        sequences = create_sequences_lon_only(group, sequence_length)
        if vid in train_vids:
            train_sequences.extend(sequences)
        elif vid in test_vids:
            test_sequences.extend(sequences)
        elif vid in val_vids:
            val_sequences.extend(sequences)

    X_train, y_train = prepare_data_lon_only(train_sequences, sequence_length)
    X_test, y_test = prepare_data_lon_only(test_sequences, sequence_length)
    X_val, y_val = prepare_data_lon_only(val_sequences, sequence_length)

    return X_train, y_train, X_test, y_test, X_val, y_val


### Splits data into train, test, and validation sets by vehicle ID

def create_train_test_val_sets_grouped_lat_only(df, sequence_length):
    vids = df['vid'].unique()
    train_vids = vids[:int(len(vids) * 0.8)]
    test_vids = vids[int(len(vids) * 0.8):int(len(vids) * 0.9)]
    val_vids = vids[int(len(vids) * 0.9):]

    train_sequences = []
    test_sequences = []
    val_sequences = []

    for vid, group in df.groupby('vid'):
        group = group.sort_values(by='ts')
        sequences = create_sequences_lat_only(group, sequence_length)
        if vid in train_vids:
            train_sequences.extend(sequences)
        elif vid in test_vids:
            test_sequences.extend(sequences)
        elif vid in val_vids:
            val_sequences.extend(sequences)

    X_train, y_train = prepare_data_lat_only(train_sequences, sequence_length)
    X_test, y_test = prepare_data_lat_only(test_sequences, sequence_length)
    X_val, y_val = prepare_data_lat_only(val_sequences, sequence_length)

    return X_train, y_train, X_test, y_test, X_val, y_val

###############################################################################

### Builds and compiles a simple RNN model for longitude prediction

def build_simple_rnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(50, activation='relu'),
        Dense(1)
    ])
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model

###############################################################################

### Builds and compiles a stacked LSTM model for longitude prediction

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, activation='relu', return_sequences=True),
        LSTM(100, activation='relu'),
        Dense(1)
    ])
    model.summary()
    optimizer = Adam(learning_rate=0.001)
    # optimizer = Adam()
    model.compile(optimizer=optimizer, loss='mse')
    return model

###############################################################################

### Defines a function to take the last time step from a tensor
### (needed to fix lambda error for loading LSTM with attention model)

def take_last_time_step(input_tensor):
    return input_tensor[:, -1, :]

###############################################################################

### Builds and compiles an LSTM model with attention for longitude prediction

def build_lstm_model_with_attention(input_shape):
    ### Longitude
    # inputs = Input(shape=input_shape)
    # lstm_out = LSTM(128, return_sequences=True)(inputs)
    
    # # Apply Attention mechanism
    # attention_output = Attention()([lstm_out, lstm_out])

    # # Directly connect attention output to a dense layer before the final output
    # dense1 = Dense(64, activation='relu')(attention_output)
    # dense2 = Dense(32, activation='relu')(dense1)

    # # Output layer (taking the last time step)
    # outputs = Dense(1)(dense2)
    # outputs = Lambda(take_last_time_step)(outputs)

    # model = Model(inputs=inputs, outputs=outputs)

    # optimizer = Adam()
    # model.compile(optimizer=optimizer, loss='mse')
    # model.summary()
    
    ### Latitude
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    
    # Apply Attention mechanism
    attention_output = Attention()([lstm_out, lstm_out])

    # Directly connect attention output to a dense layer before the final output
    dense1 = Dense(32, activation='relu')(attention_output)

    # Output layer (taking the last time step)
    outputs = Dense(1)(dense1)
    outputs = Lambda(take_last_time_step)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = Adam(clipvalue=1, learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model

###############################################################################

### Trains the model and plots training/validation loss over epochs with early stopping

def plot_and_validate_model_lon_only(model, X_train, y_train, X_val, y_val, model_name, epochs, batch_size):
    """Trains and plots training and validation loss for longitude prediction."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    epochs_trained = len(history.history['loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')

    num_epochs = len(history.history['loss'])
    if num_epochs <= 20:
        plt.xticks(np.arange(0, num_epochs, 1))
    elif num_epochs <= 100:
        plt.xticks(np.arange(0, num_epochs, 5))
    elif num_epochs <= 500:
        plt.xticks(np.arange(0, num_epochs, 25))
    elif num_epochs <=1000:
        plt.xticks(np.arange(0, num_epochs, 50))
    else:
        plt.xticks(np.arange(0, num_epochs, 100))

    plt.title(f'{model_name} - Training and Validation Loss (Longitude)\nEpochs Trained: {epochs_trained}')
    plt.legend()
    plt.show()

    return model, epochs_trained

### Trains the model and plots training/validation loss over epochs with early stopping

def plot_and_validate_model_lat_only(model, X_train, y_train, X_val, y_val, model_name, epochs, batch_size):
    """Trains and plots training and validation loss for latitude prediction."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    epochs_trained = len(history.history['loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')

    # Optimize x-axis ticks
    num_epochs = len(history.history['loss'])
    if num_epochs <= 20:
        plt.xticks(np.arange(0, num_epochs, 1))  # Show every epoch
    elif num_epochs <= 100:
        plt.xticks(np.arange(0, num_epochs, 5))  # Show every 5 epochs
    elif num_epochs <= 500:
        plt.xticks(np.arange(0, num_epochs, 25)) #show every 25 epochs
    elif num_epochs <=1000:
        plt.xticks(np.arange(0, num_epochs, 50)) #show every 50 epochs
    else:
        plt.xticks(np.arange(0, num_epochs, 100)) # show every 100 epochs

    plt.title(f'{model_name} - Training and Validation Loss (Latitude)\nEpochs Trained: {epochs_trained}')
    plt.legend()
    plt.show()

    return model, epochs_trained

###############################################################################

### Plots actual versus predicted values for a subset of the test data

def plot_predictions(y_test, y_pred, model_name, n_samples=100):
    """Plots actual vs. predicted values for a subset of the test data."""
    y_test_np = np.array(y_test).flatten()
    y_pred_np = np.array(y_pred).flatten()

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(n_samples), y_test_np[:n_samples], label='Actual', marker='o', linestyle='-')
    plt.plot(np.arange(n_samples), y_pred_np[:n_samples], label='Predicted', marker='x', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Longitude/Latitude Value')
    plt.title(f'{model_name} - Actual vs. Predicted Values (First {n_samples} Samples)')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################

### Plots the errors (residuals) between predicted and actual values.
    
def plot_residuals(y_test, y_pred, model_name):
    """Plots the residuals (errors) of the model's predictions."""
    y_test_np = np.array(y_test).flatten()
    y_pred_np = np.array(y_pred).flatten()
    residuals = y_pred_np - y_test_np

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_np, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Value')
    plt.ylabel('Residuals (Predicted - Actual)')
    plt.title(f'{model_name} - Residual Plot')
    plt.grid(True)
    plt.show()

###############################################################################

### Autoregressively predicts and plots the future trajectory of a single vehicle

def plot_actual_vs_predicted_with_future(model, df, vid, sequence_length, coordinate_column='lon', n_predictions=50):
    """Predicts the future trajectory and plots actual past, actual future, and predicted future."""
    vehicle_data = df[df['vid'] == vid].sort_values(by='ts')[coordinate_column].values
    if len(vehicle_data) < sequence_length + n_predictions:
        print(f"Not enough data for vehicle {vid} to show {sequence_length} past and {n_predictions} future actual values.")
        return

    last_sequence = vehicle_data[-sequence_length:].reshape(1, sequence_length, 1)
    predicted_future = []
    for _ in range(n_predictions):
        next_pred = model.predict(last_sequence)[0, 0]
        predicted_future.append(next_pred)
        last_sequence = np.concatenate([last_sequence[:, 1:, :], [[[next_pred]]]], axis=1)

    past_trajectory = vehicle_data[-sequence_length:]
    actual_future = vehicle_data[-n_predictions:]  # Get the actual future values
    future_time_steps = np.arange(len(past_trajectory), len(past_trajectory) + n_predictions)
    past_time_steps = np.arange(len(past_trajectory))

    plt.figure(figsize=(12, 6))
    plt.plot(past_time_steps, past_trajectory, label=f'Actual Past (Vehicle {vid})', marker='o', linestyle='-')
    plt.plot(future_time_steps, actual_future, label=f'Actual Future (Vehicle {vid})', marker='.', linestyle='-', color='green')
    plt.plot(future_time_steps, predicted_future, label=f'Predicted Future (Vehicle {vid})', marker='x', linestyle='--')
    plt.xlabel('Time Step (Relative)')
    plt.ylabel(coordinate_column.capitalize() + ' Value')
    plt.title(f'Actual vs. Predicted Trajectory for Vehicle {vid}')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################

### Plots actual versus predicted trajectory for a specific vehicle

def plot_single_vehicle_predictions(model, df, vid, sequence_length, coordinate_column='lon'):
    """Plots actual vs. predicted values for a specific vehicle."""
    vehicle_data = df[df['vid'] == vid].sort_values(by='ts')[coordinate_column].values

    if len(vehicle_data) < sequence_length:
        print(f"Not enough data for vehicle {vid} to form a sequence of length {sequence_length}.")
        return

    # Create sequences and targets for the specific vehicle
    sequences = []
    targets = []
    for i in range(len(vehicle_data) - sequence_length):
        sequences.append(vehicle_data[i:i + sequence_length])
        targets.append(vehicle_data[i + sequence_length])

    X = np.array(sequences)
    y_actual = np.array(targets)
    X = X.reshape(X.shape[0], sequence_length, 1)

    # Make predictions for this vehicle
    y_predicted = model.predict(X).flatten()

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(y_actual)), y_actual, label=f'Actual (Vehicle {vid})', marker='o', linestyle='-')
    plt.plot(np.arange(len(y_predicted)), y_predicted, label=f'Predicted (Vehicle {vid})', marker='x', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel(coordinate_column.capitalize() + ' Value')
    plt.title(f'Actual vs. Predicted {coordinate_column.capitalize()} for Vehicle {vid}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
###############################################################################

### Evaluates the model, prints metrics, and plots actual versus predicted values

def evaluate_and_plot_model(model, X_test, y_test, sequence_length, model_name, plot_n_samples=50):
    y_pred = model.predict(X_test)
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)

    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    r_squared = r2_score(y_test_np, y_pred_np)

    print(f"Evaluation Metrics for {model_name}:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R-squared (R²): {r_squared:.4f}")
    print("-" * 30)

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(y_test_np[:plot_n_samples])), y_test_np[:plot_n_samples], label='Actual', marker='o')
    plt.plot(np.arange(len(y_pred_np[:plot_n_samples])), y_pred_np[:plot_n_samples], label='Predicted', marker='x')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Longitude/Latitude Value')
    plt.title(f'{model_name} - Actual vs. Predicted Values\nSequence Length: {sequence_length}')
    plt.show()

    return y_pred_np

###############################################################################
################################ Parameters 1 #################################
###############################################################################

##### Parameters (change these to your liking)
# Keep in mind the sequence length in which your chosen model is trained and
# always use the same

sequence_length = 20
input_shape = (sequence_length, 1)
epochs = 150  # Initial epochs
batch_size = 256


###############################################################################
############################### Main execution ################################
###############################################################################

path = "C:\\School_2024-2025\\Internship\\Task3\\PROBE-202410\\20241001.csv.out"
df = pd.read_csv(path)
transformed_df = transform_taxi_data(df)

# X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val_sets_grouped_lon_only(transformed_df, sequence_length)
X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val_sets_grouped_lat_only(transformed_df, sequence_length)

# transformed_df.info()
# vid_count  = transformed_df['vid'].value_counts()

###############################################################################
################################## Training ###################################
###############################################################################

##### Build, train, and save models for longitude
    # (Only run to train new model)

### RNN Model
# rnn_model_lon = build_simple_rnn_model(input_shape)
# trained_rnn_model_lon, rnn_epochs_trained_lon = plot_and_validate_model_lon_only(
#     rnn_model_lon, X_train, y_train, X_val, y_val, "RNN Model (Longitude)", epochs, batch_size
# )
# trained_rnn_model_lon.save('rnn_model_lon_only.keras')
### LSTM Model
# lstm_model_lon = build_lstm_model(input_shape)
# trained_lstm_model_lon, lstm_epochs_trained_lon = plot_and_validate_model_lon_only(
#     lstm_model_lon, X_train, y_train, X_val, y_val, "LSTM Model (Longitude)", epochs, batch_size
# )
# trained_lstm_model_lon.save('lstm_model_lon_only.keras')
### LSTM with attention Model
# lstm_attention_model_lon = build_lstm_model_with_attention(input_shape)
# trained_lstm_attention_model_lon, lstm_attention_epochs_trained_lon = plot_and_validate_model_lon_only(
#     lstm_attention_model_lon, X_train, y_train, X_val, y_val, "LSTM with Attention Model (Longitude)", epochs, batch_size
# )
# trained_lstm_attention_model_lon.save('lstm_attention_model_lon_only.keras')

##### Build, train, and save models for latitude
    # (Only run to train new model)

### RNN Model
# rnn_model_lat = build_simple_rnn_model(input_shape)
# trained_rnn_model_lat, rnn_epochs_trained_lat = plot_and_validate_model_lat_only(
#     rnn_model_lat, X_train, y_train, X_val, y_val, "RNN Model (Latitude)", epochs, batch_size
# )
# trained_rnn_model_lat.save('rnn_model_lat_only.keras')
### LSTM Model
# lstm_model_lat = build_lstm_model(input_shape)
# trained_lstm_model_lat, lstm_epochs_trained_lon = plot_and_validate_model_lat_only(
#     lstm_model_lat, X_train, y_train, X_val, y_val, "LSTM Model (Latitude)", epochs, batch_size
# )
# trained_lstm_model_lat.save('lstm_model_lat_only.keras')
### LSTM with attention Model
# lstm_attention_model_lat = build_lstm_model_with_attention(input_shape)
# trained_lstm_attention_model_lat, lstm_attention_epochs_trained_lat = plot_and_validate_model_lat_only(
#     lstm_attention_model_lat, X_train, y_train, X_val, y_val, "LSTM with Attention Model (Latitude)", epochs, batch_size
# )
# trained_lstm_attention_model_lat.save('lstm_attention_model_lat_only.keras')

###############################################################################
################################ Loading model ################################
###############################################################################

##### Load the pre-trained models for longitude

### /// Places where I stored the models to evaluate
      # rnn_model_lon_only.keras
      # lstm_model_lon_only.keras
      # lstm_attention_model_lon_only.keras
      # LongSeq50/rnn_model_lon_only.keras
      # FinalLongitudeModels&Plots/rnn_model_lon_only.keras
      # FinalLongitudeModels&Plots/lstm_model_lon_only.keras
      # FinalLongitudeModels&Plots/lstm_attention_model_lon_only.keras
      
# RNN Model
# trained_rnn_model_lon = keras.models.load_model('rnn_model_lon_only.keras')
# LSTM Model
# trained_lstm_model_lon = keras.models.load_model('FinalLongitudeModels&Plots/lstm_model_lon_only.keras')
# LSTM with attention Model
keras.config.enable_unsafe_deserialization()
trained_lstm_attention_model_lon = keras.models.load_model(
    'FinalLongitudeModels&Plots/lstm_attention_model_lon_only.keras',
    custom_objects={'take_last_time_step': take_last_time_step}
)

##### Load the pre-trained models for latitude

### /// Places where I stored the models to evaluate
      # rnn_model_lat_only.keras
      # lstm_model_lat_only.keras
      # lstm_attention_model_lat_only.keras
      # FinalLatitudeModels&Plots/rnn_model_lat_only.keras
      # FinalLatitudeModels&Plots/lstm_model_lat_only.keras
      # FinalLatitudeModels&Plots/lstm_attention_model_lat_only.keras
      
# RNN Model
# trained_rnn_model_lat = keras.models.load_model('FinalLatitudeModels&Plots/rnn_model_lat_only.keras')
# LSTM Model
# trained_lstm_model_lat = keras.models.load_model('FinalLatitudeModels&Plots/lstm_model_lat_only.keras')
# LSTM with attention Model
# keras.config.enable_unsafe_deserialization()
# trained_lstm_attention_model_lat = keras.models.load_model(
#     'FinalLatitudeModels&Plots/lstm_attention_model_lat_only.keras',
#     custom_objects={'take_last_time_step': take_last_time_step}
# )

###############################################################################
################################ Parameters 2 #################################
###############################################################################

##### Parameters (change these to your liking)
# Keep in mind the sequence length in which your chosen model is trained and
# always use the same

n_samples=150

### /// Choose the name of the model to evaluate
      # RNN Model
      # LSTM Model
      # LSTM with attention Model

model_name = 'LSTM with attention Model'

### /// Choose which model to evaluate
      # trained_rnn_model_lon
      # trained_lstm_model_lon
      # trained_lstm_attention_model_lon
      # trained_rnn_model_lat
      # trained_lstm_model_lat
      # trained_lstm_attention_model_lat

model_to_evaluate = trained_lstm_attention_model_lon

### /// Vehicle ids to test
      # V8OW60TLKR4j/ZWUSqhG+9Bw4TM
      # u8H8BYM25mnEmhPv+pDdg4aKPd8
      
vehicle_id = 'V8OW60TLKR4j/ZWUSqhG+9Bw4TM'

### /// Column to test 
      # lon
      # lat
      
coordinate_column='lon'

###############################################################################
################################# Evaluation ##################################
###############################################################################

##### Predict

y_pred = model_to_evaluate.predict(X_test)
y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)

### Predict the first 20 values
if len(y_test_np) == len(y_pred_np):
    print(f"\nFirst 20 Actual vs. Predicted Values ({coordinate_column}):")
    for i in range(min(20, len(y_test_np))):
        print(f"Actual: {y_test_np[i][0]:.5f}, Predicted: {y_pred_np[i][0]:.5f}")
else:
    print("Error: Actual and predicted arrays have different lengths.")

###############################################################################

##### Evaluate

# Example Usage (assuming you have your model, X_test, y_test, and sequence_length defined):
y_predicted = evaluate_and_plot_model(model_to_evaluate, X_test, y_test, sequence_length, f"{model_name} ({coordinate_column})")

### Visualizing Predictions Against Actual Values (More Detailed):
plot_predictions(y_test, y_predicted, f"{model_name} ({coordinate_column})", n_samples)

### Plotting the residuals
plot_residuals(y_test, y_predicted, f"{model_name} ({coordinate_column})")

### Predictions with actual sequences
# Overal model

# Specific vehicle
plot_single_vehicle_predictions(model_to_evaluate, transformed_df, vehicle_id, sequence_length, coordinate_column)

### Autoregressive predictions of a specific vehicle
plot_actual_vs_predicted_with_future(model_to_evaluate, transformed_df, vehicle_id, sequence_length, coordinate_column, n_predictions=100)


