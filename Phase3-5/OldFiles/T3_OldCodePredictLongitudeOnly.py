# pip install tensorflow_addons
# pip install --upgrade tensorflow keras

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input, Attention
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

def create_train_test_val_sets_grouped_lon_only(df, sequence_length):
    """Creates train, test, and validation sets for longitude prediction,
    grouped by 'vid' and chronologically."""

    vids = df['vid'].unique()
    train_vids = vids[:int(len(vids) * 0.8)]
    test_vids = vids[int(len(vids) * 0.8):int(len(vids) * 0.9)]
    val_vids = vids[int(len(vids) * 0.9):]

    train_sequences = []
    test_sequences = []
    val_sequences = []

    for vid, group in df.groupby('vid'):
        group = group.sort_values(by='ts')  # Ensure chronological order
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

# ------------------------------------------------------------------------------

# we create the sequence - MODIFIED FOR LONGITUDE ONLY
def create_sequences_lon_only(df, sequence_length):
    """Creates input sequences for longitude prediction from DataFrame."""
    sequences = []
    for vid, group in df.groupby('vid'):
        lons = group['lon'].tolist()
        for i in range(len(lons) - sequence_length):
            seq_lons = lons[i:i + sequence_length]
            target_lon = lons[i + sequence_length]
            sequences.append(seq_lons)
    return sequences

# ------------------------------------------------------------------------------

# we need the data to be prepared before the training of the model - MODIFIED FOR LONGITUDE ONLY
# we first store the sequences
# then convert the X and y to a NumPy array (correct shape for RNN/LSTM)
def prepare_data_lon_only(sequences, sequence_length):
    """Prepares data for RNN/LSTM model for longitude prediction."""
    X = np.array(sequences)
    y = np.array([seq[-1] for seq in sequences])  # The last element of each sequence is the target lon
    X = np.reshape(X, (X.shape[0], sequence_length, 1)) # Reshape for single feature (longitude)
    y = np.reshape(y, (-1, 1))
    return X, y

# ------------------------------------------------------------------------------

# to build a RNN model - MODIFIED FOR LONGITUDE ONLY
# change the dense layers, activation function, optimizer or loss
def build_simple_rnn_model_lon_only(input_shape):
    """Builds SimpleRNN model for longitude prediction."""
    model = Sequential([
        Input(shape=input_shape),  # Use Input layer as first layer
        SimpleRNN(50, activation='relu'),
        Dense(1)  # Output only one value (longitude)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------------------------------------------------------------

# to build a LSTM model - MODIFIED FOR LONGITUDE ONLY
# change the dense layers, activation function, optimizer or loss
def build_lstm_model_lon_only(input_shape):
    """Builds LSTM model for longitude prediction."""
    model = Sequential([
        Input(shape=input_shape),  # Use Input layer as first layer
        LSTM(50, activation='relu'),
        Dense(1)  # Output only one value (longitude)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ------------------------------------------------------------------------------

###### still add dropout like this one: model.add(LSTM(100, activation='relu', return_sequences=True, dropout=0.2))

# to build a LSTM model with attention - MODIFIED FOR LONGITUDE ONLY
def build_lstm_model_with_attention_lon_only(input_shape):
    """Builds LSTM model with attention for longitude prediction."""
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
    outputs = Dense(1)(dense2)  # Output only one value (longitude)
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

# ------------------------------------------------------------------------------

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
    plt.ylim(0, 0.02)
    plt.xlim(0, len(history.history['loss']))

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

    plt.title(f'{model_name} - Training and Validation Loss (Longitude)\nEpochs Trained: {epochs_trained}')
    plt.legend()
    plt.show()

    return model, epochs_trained

# ------------------------------------------------------------------------------

def evaluate_and_plot_optimized_models_lon_only(model, X_test, y_test, sequence_length, model_name):
    """Evaluates and plots predictions from an optimized model for longitude."""
    y_pred = model.predict(X_test)
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    print(f"{model_name} MAE (Longitude): {mae}")
    print(f"{model_name} RMSE (Longitude): {rmse}")

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(y_test_np[:100])), y_test_np[:100], label='Actual Longitude', marker='o')
    plt.plot(np.arange(len(y_pred_np[:100])), y_pred_np[:100], label='Predicted Longitude', marker='x')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Longitude')
    plt.title(f'{model_name} - Actual vs. Predicted Longitude\nSequence Length: {sequence_length}')
    plt.show()

# ------------------------------------------------------------------------------

# Main execution
url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(url)
transformed_df = transform_taxi_data(df)

sequence_length = 20

# creating the datasets - MODIFIED FOR LONGITUDE ONLY
X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val_sets_grouped_lon_only(transformed_df, sequence_length)

# variables for training the model - MODIFIED FOR LONGITUDE ONLY INPUT SHAPE
input_shape = (sequence_length, 1)
epochs = 100  # Initial epochs
batch_size = 32

# Build, train, and plot losses for RNN
rnn_model = build_simple_rnn_model_lon_only(input_shape)
trained_rnn_model, rnn_epochs_trained = plot_and_validate_model_lon_only(
    rnn_model, X_train, y_train, X_val, y_val, "RNN Model (Longitude)", epochs, batch_size
)
trained_rnn_model.save('rnn_model_lon_only.keras')
evaluate_and_plot_optimized_models_lon_only(trained_rnn_model, X_test, y_test, sequence_length, "RNN Model")

###########################################

# Build, train, and plot losses for LSTM
lstm_model = build_lstm_model_lon_only(input_shape)
trained_lstm_model, lstm_epochs_trained = plot_and_validate_model_lon_only(
    lstm_model, X_train, y_train, X_val, y_val, "LSTM Model (Longitude)", epochs, batch_size
)
trained_lstm_model.save('lstm_model_lon_only.keras')
evaluate_and_plot_optimized_models_lon_only(trained_lstm_model, X_test, y_test, sequence_length, "LSTM Model")

##########################################

# Build, train, and plot losses for LSTM with Attention
lstm_attention_model = build_lstm_model_with_attention_lon_only(input_shape)
trained_lstm_attention_model, lstm_attention_epochs_trained = plot_and_validate_model_lon_only(
    lstm_attention_model, X_train, y_train, X_val, y_val, "LSTM with Attention Model (Longitude)", epochs, batch_size
)
trained_lstm_attention_model.save('lstm_attention_model_lon_only.keras')
evaluate_and_plot_optimized_models_lon_only(trained_lstm_attention_model, X_test, y_test, sequence_length, "LSTM with Attention Model")