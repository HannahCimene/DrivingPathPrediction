#pip install tensorflow_addons
#pip install --upgrade tensorflow keras

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input, Attention
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def create_train_test_val_sets_grouped(df, sequence_length):
    """Creates train, test, and validation sets from DataFrame, grouped by 'vid' and chronologically."""

    vids = df['vid'].unique()
    train_vids = vids[:int(len(vids) * 0.8)]
    test_vids = vids[int(len(vids) * 0.8):int(len(vids) * 0.9)]
    val_vids = vids[int(len(vids) * 0.9):]

    train_sequences = []
    test_sequences = []
    val_sequences = []

    for vid, group in df.groupby('vid'):
        group = group.sort_values(by='ts')  # Ensure chronological order
        sequences = create_sequences(group, sequence_length)  # Use your existing create_sequences
        if vid in train_vids:
            train_sequences.extend(sequences)
        elif vid in test_vids:
            test_sequences.extend(sequences)
        elif vid in val_vids:
            val_sequences.extend(sequences)

    X_train, y_train = prepare_data(train_sequences, sequence_length)
    X_test, y_test = prepare_data(test_sequences, sequence_length)
    X_val, y_val = prepare_data(val_sequences, sequence_length)

    return X_train, y_train, X_test, y_test, X_val, y_val

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

# to build a RNN model
# change the dense layers, activation function, optimizer or loss
def build_simple_rnn_model(input_shape):
    """Builds SimpleRNN model."""
    model = Sequential([
        Input(shape=input_shape),  # Use Input layer as first layer
        SimpleRNN(50, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

#------------------------------------------------------------------------------

# to build a LSTM model
# change the dense layers, activation function, optimizer or loss
def build_lstm_model(input_shape):
    """Builds LSTM model."""
    model = Sequential([
        Input(shape=input_shape),  # Use Input layer as first layer
        LSTM(50, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

def train_and_plot_losses(X_train, y_train, X_test, y_test, X_val, y_val, input_shape, epochs, batch_size, sequence_length):
    """Trains RNN, LSTM, and LSTM with attention, plots losses, and saves models."""

    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,         # Stop after 10 epochs with no improvement
        restore_best_weights=True,  # Restore model weights from the epoch with the best validation loss
        verbose=1 #added verbose to print when early stopping occurs.
    )

    # RNN Model
    rnn_model = build_simple_rnn_model(input_shape)
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),  # Add validation data
        callbacks=[early_stopping]  # Add EarlyStopping callback
    )
    rnn_model.save('rnn_model.keras')
    rnn_epochs_trained = len(rnn_history.history['loss'])
    print(f"RNN Model stopped training after {rnn_epochs_trained} epochs.")

    # LSTM Model
    lstm_model = build_lstm_model(input_shape)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val), # Add validation data
        callbacks=[early_stopping] # Add EarlyStopping callback
    )
    lstm_model.save('lstm_model.keras')
    lstm_epochs_trained = len(lstm_history.history['loss'])
    print(f"LSTM Model stopped training after {lstm_epochs_trained} epochs.")

    # LSTM with Attention Model
    lstm_attention_model = build_lstm_model_with_attention(input_shape)
    lstm_attention_history = lstm_attention_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val), # Add validation data
        callbacks=[early_stopping] # Add EarlyStopping callback
    )
    lstm_attention_model.save('lstm_attention_model.keras')
    lstm_attention_epochs_trained = len(lstm_attention_history.history['loss'])
    print(f"LSTM with Attention Model stopped training after {lstm_attention_epochs_trained} epochs.")

    # Plotting Losses
    plt.figure(figsize=(12, 6))
    plt.plot(rnn_history.history['loss'], label='RNN Loss')
    plt.plot(lstm_history.history['loss'], label='LSTM Loss')
    plt.plot(lstm_attention_history.history['loss'], label='LSTM with Attention Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Losses')
    plt.legend()
    plt.show()

    return rnn_epochs_trained, lstm_epochs_trained, lstm_attention_epochs_trained

#------------------------------------------------------------------------------

# def plot_and_validate_model(history, model_name, epochs_trained):
#     """Plots training and validation loss."""
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss (MSE)')
#     now = datetime.datetime.now()
#     date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
#     plt.title(f'{model_name} - Training and Validation Loss\nDate/Time: {date_time_str}\nEpochs Trained: {epochs_trained}')
#     plt.legend()
#     plt.show()

def plot_and_validate_model(history, model_name, epochs_trained):
    """Plots training and validation loss with optimized x-axis ticks."""
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

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    plt.title(f'{model_name} - Training and Validation Loss\nDate/Time: {date_time_str}\nEpochs Trained: {epochs_trained}')
    plt.legend()
    plt.show()
    
#------------------------------------------------------------------------------
    
def evaluate_and_plot_optimized_models(X_test, y_test, sequence_length):
    """Evaluates and plots predictions from optimized models."""
    loaded_rnn_model = tf.keras.models.load_model('rnn_model.keras')
    loaded_lstm_model = tf.keras.models.load_model('lstm_model.keras')
    tf.keras.config.enable_unsafe_deserialization()
    loaded_attention_model = tf.keras.models.load_model('lstm_attention_model.keras')

    models = {"RNN Model": loaded_rnn_model, "LSTM Model": loaded_lstm_model, "LSTM with Attention Model": loaded_attention_model}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_test_np = np.array(y_test)
        y_pred_np = np.array(y_pred)
        mae = mean_absolute_error(y_test_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
        print(f"{model_name} MAE: {mae}")
        print(f"{model_name} RMSE: {rmse}")

        plt.figure(figsize=(15, 6))
        plt.plot(y_test_np[:100, 0], y_test_np[:100, 1], label='Actual', marker='o')
        plt.plot(y_pred_np[:100, 0], y_pred_np[:100, 1], label='Predicted', marker='x')
        plt.legend()
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        plt.title(f'{model_name} - Actual vs. Predicted Trajectories\nDate/Time: {date_time_str}\nSequence Length: {sequence_length}')
        plt.show()

#------------------------------------------------------------------------------

# Main execution
url = 'https://raw.githubusercontent.com/Rathachai/DA101/refs/heads/gh-pages/data/gps-data.csv'
df = pd.read_csv(url)
# print("Original DataFrame Info:")
# df.info()
transformed_df = transform_taxi_data(df)
print("\nFinal Transformed DataFrame Info:")
transformed_df.info()
print("\nFinal Transformed DataFrame Head:")
print(transformed_df.head())

sequence_length=20

sequences = create_sequences(transformed_df, sequence_length)

# creating the datasets
X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val_sets_grouped(transformed_df, sequence_length)

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)
# print("X_val shape:", X_val.shape)
# print("y_val shape:", y_val.shape)

# variables for training the model
input_shape = (sequence_length, 2)
epochs = 100  # Initial epochs (not used directly for final training)
batch_size = 32

rnn_epochs_trained, lstm_epochs_trained, lstm_attention_epochs_trained = train_and_plot_losses(X_train, y_train, X_test, y_test, X_val, y_val, input_shape, epochs, batch_size, sequence_length)

rnn_model = build_simple_rnn_model(input_shape)
rnn_history = rnn_model.fit(X_train, y_train, epochs=rnn_epochs_trained, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)])
rnn_model.save('rnn_model.keras')
lstm_model = build_lstm_model(input_shape)
lstm_history = lstm_model.fit(X_train, y_train, epochs=lstm_epochs_trained, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)])
lstm_model.save('lstm_model.keras')
lstm_attention_model = build_lstm_model_with_attention(input_shape)
lstm_attention_history = lstm_attention_model.fit(X_train, y_train, epochs=lstm_attention_epochs_trained, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)])
lstm_attention_model.save('lstm_attention_model.keras')

plot_and_validate_model(rnn_history, "RNN Model", rnn_epochs_trained)
plot_and_validate_model(lstm_history, "LSTM Model", lstm_epochs_trained)
plot_and_validate_model(lstm_attention_history, "LSTM with Attention Model", lstm_attention_epochs_trained)

evaluate_and_plot_optimized_models(X_test, y_test, sequence_length)