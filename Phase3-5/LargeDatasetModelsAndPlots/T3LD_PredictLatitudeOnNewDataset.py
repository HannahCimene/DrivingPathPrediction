import keras
import numpy as np
import pandas as pd

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
    train_vids = vids[:int(len(vids) * 0.1)]
    test_vids = vids[int(len(vids) * 0.1):int(len(vids) * 0.9)]
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

path = "C:\\School_2024-2025\\Internship\\Task3\\PROBE-202410\\20241001.csv.out"
df = pd.read_csv(path)
transformed_df = transform_taxi_data(df)

transformed_df.info()

vid_count  = transformed_df['vid'].value_counts()

sequence_length = 20

X_train, y_train, X_test, y_test, X_val, y_val = create_train_test_val_sets_grouped(transformed_df, sequence_length)

###############################################################################

model = keras.models.load_model('lstm_attention_model_lat_only.keras')

y_pred = model.predict(X_test)
y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)

# print actual and predicted values
if len(y_test_np) == len(y_pred_np):
    print("\nFirst 20 Actual vs. Predicted LSTM attention Values:")
    for i in range(min(20, len(y_test_np))): # Print the first 20 for brevity
        print(f"Actual: {y_test_np[i][0]:.5f}, Predicted: {y_pred_np[i][0]:.5f}")
else:
    print("Error: Actual and predicted arrays have different lengths.")