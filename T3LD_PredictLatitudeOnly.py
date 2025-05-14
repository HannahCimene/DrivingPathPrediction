# pip install tensorflow_addons
# pip install --upgrade tensorflow keras

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input, Attention, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

###############################################################################

def transform_taxi_data(df):
    """Transforms taxi data and removes specified vids."""
    # remove duplicates
    df.columns = ['vid', 'valgps', 'lat', 'lon', 'ts', 'speed', 'dir', 'hirelight', 'engineactive']
    df = df.drop_duplicates(subset=['ts', 'lat', 'lon'])
    # remove speed 0 and speed above 125
    df = df[(df['speed'] > 0) & (df['speed'] <= 125)].copy()
    # remove the non valid gps points
    df = df[df['valgps'] == 1]
    # remove the data where there are no passengers in the taxi
    df = df[df['hirelight'] == 0]
    # remove the vehicles with less then 15 datapoints
    vehicle_counts = df['vid'].value_counts()
    
    ### change the count to more then the sequence
    vehicles_to_remove = vehicle_counts[vehicle_counts < 15].index
    df = df[~df['vid'].isin(vehicles_to_remove)]

    # change the timestamp to datetime
    df['ts'] = pd.to_datetime(df['ts'])
    df = df[['vid', 'ts', 'lat', 'lon']].copy()
    df = df.sort_values(by='ts')

    df = df.reset_index(drop=True)

    return df

###############################################################################

def create_sequences_lat_only(df, sequence_length):
    sequences = []
    for vid, group in df.groupby('vid'):
        lats = group['lat'].tolist()
        for i in range(len(lats) - sequence_length):
            seq_lats = lats[i:i + sequence_length]
            target_lat = lats[i + sequence_length]
            sequences.append(seq_lats)
    return sequences

###############################################################################

def prepare_data_lat_only(sequences, sequence_length):
    X = np.array(sequences)
    y = np.array([seq[-1] for seq in sequences])
    X = np.reshape(X, (X.shape[0], sequence_length, 1))
    y = np.reshape(y, (-1, 1))
    return X, y

###############################################################################

def create_train_test_val_sets_grouped(df, sequence_length):
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

# to build a RNN model

def build_simple_rnn_model_lat_only(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(50, activation='relu'),
        Dense(1)
    ])
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model

###############################################################################

# to build a LSTM model

def build_lstm_model_lat_only(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(100, activation='relu'),
        Dense(1)
    ])
    model.summary()
    # optimizer = Adam(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss='mse')
    model.compile(optimizer='adam', loss='mse')

    return model

###############################################################################

###### still add dropout like this one: model.add(LSTM(100, activation='relu', return_sequences=True, dropout=0.2))

# to build a LSTM model with attention - MODIFIED FOR LATITUDE ONLY
def build_lstm_model_with_attention_lat_only(input_shape):
    """Builds a simplified LSTM model with attention for latitude prediction."""
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(128, return_sequences=True)(inputs) # Increased LSTM units slightly

    # Apply Attention mechanism
    attention_output = Attention()([lstm_out, lstm_out]) # query and value are both lstm_out

    # Directly connect attention output to a dense layer before the final output
    dense1 = Dense(32, activation='relu')(attention_output)

    # Output layer (taking the last time step)
    outputs = Dense(1)(dense1)
    outputs = Lambda(lambda x: x[:, -1, :])(outputs)

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

def evaluate_and_plot_optimized_models_lat_only(model, X_test, y_test, sequence_length, model_name):
    """Evaluates and plots predictions from an optimized model for latitude."""
    y_pred = model.predict(X_test)
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    print(f"{model_name} MAE (Latitude): {mae}")
    print(f"{model_name} RMSE (Latitude): {rmse}")

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(y_test_np[:50])), y_test_np[:50], label='Actual Latitude', marker='o')
    plt.plot(np.arange(len(y_pred_np[:50])), y_pred_np[:50], label='Predicted Latitude', marker='x')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Latitude')
    plt.title(f'{model_name} - Actual vs. Predicted Latitude\nSequence Length: {sequence_length}')
    plt.show()

###############################################################################

# Main execution

path = "C:\\School_2024-2025\\Internship\\Task3\\PROBE-202410\\20241001.csv.out"
df = pd.read_csv(path)
transformed_df = transform_taxi_data(df)

transformed_df.info()

vid_count  = transformed_df['vid'].value_counts()

sequence_length = 20

X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val_sets_grouped(transformed_df, sequence_length)

input_shape = (sequence_length, 1)
epochs = 150  # Initial epochs
batch_size = 256

###############################################################################

### Build, train, and plot losses for RNN

# rnn_model = build_simple_rnn_model_lat_only(input_shape)
# trained_rnn_model, rnn_epochs_trained = plot_and_validate_model_lat_only(
#     rnn_model, X_train, y_train, X_val, y_val, "RNN Model (Latitude)", epochs, batch_size
# )
# trained_rnn_model.save('rnn_model_lat_only.keras')
# evaluate_and_plot_optimized_models_lat_only(trained_rnn_model, X_test, y_test, sequence_length, "RNN Model")

# ### Evaluate the RNN model

# model = keras.models.load_model('rnn_model_lat_only.keras')

# y_pred = model.predict(X_test)
# y_test_np = np.array(y_test)
# y_pred_np = np.array(y_pred)

# # print actual and predicted values
# if len(y_test_np) == len(y_pred_np):
#     print("\nFirst 20 Actual vs. Predicted RNN Values:")
#     for i in range(min(20, len(y_test_np))): # Print the first 20 for brevity
#         print(f"Actual: {y_test_np[i][0]:.5f}, Predicted: {y_pred_np[i][0]:.5f}")
# else:
#     print("Error: Actual and predicted arrays have different lengths.")

###############################################################################

### Build, train, and plot losses for LSTM

lstm_model = build_lstm_model_lat_only(input_shape)
trained_lstm_model, lstm_epochs_trained = plot_and_validate_model_lat_only(
    lstm_model, X_train, y_train, X_val, y_val, "LSTM Model (Latitude)", epochs, batch_size
)
trained_lstm_model.save('lstm_model_lat_only.keras')
evaluate_and_plot_optimized_models_lat_only(trained_lstm_model, X_test, y_test, sequence_length, "LSTM Model")

### Evaluate the LSTM model

model = keras.models.load_model('lstm_model_lat_only.keras')

y_pred = model.predict(X_test)
y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)

# print actual and predicted values
if len(y_test_np) == len(y_pred_np):
    print("\nFirst 20 Actual vs. Predicted LSTM Values:")
    for i in range(min(20, len(y_test_np))): # Print the first 20 for brevity
        print(f"Actual: {y_test_np[i][0]:.5f}, Predicted: {y_pred_np[i][0]:.5f}")
else:
    print("Error: Actual and predicted arrays have different lengths.")
    
###############################################################################

### Build, train, and plot losses for LSTM with Attention

# lstm_attention_model = build_lstm_model_with_attention_lat_only(input_shape)
# trained_lstm_attention_model, lstm_attention_epochs_trained = plot_and_validate_model_lat_only(
#     lstm_attention_model, X_train, y_train, X_val, y_val, "LSTM with Attention Model (Latitude)", epochs, batch_size
# )
# trained_lstm_attention_model.save('lstm_attention_model_lat_only.keras')
# evaluate_and_plot_optimized_models_lat_only(trained_lstm_attention_model, X_test, y_test, sequence_length, "LSTM with Attention Model")

# ### Evaluate the LSTM with attention model

# keras.config.enable_unsafe_deserialization()
# model = keras.models.load_model('lstm_attention_model_lat_only.keras')

# y_pred = model.predict(X_test)
# y_test_np = np.array(y_test)
# y_pred_np = np.array(y_pred)

# # print actual and predicted values
# if len(y_test_np) == len(y_pred_np):
#     print("\nFirst 20 Actual vs. Predicted LSTM attention Values:")
#     for i in range(min(20, len(y_test_np))): # Print the first 20 for brevity
#         print(f"Actual: {y_test_np[i][0]:.5f}, Predicted: {y_pred_np[i][0]:.5f}")
# else:
#     print("Error: Actual and predicted arrays have different lengths.")







