import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt

pd.options.display.max_rows = 10

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df = train_df.reindex(np.random.permutation(train_df.index))
train_df["label"] = (train_df["label"] == "correct").astype(float)


train_df['pickup_time'] = pd.to_datetime(train_df['pickup_time'])
test_df['pickup_time'] = pd.to_datetime(test_df['pickup_time'])

train_df['pickup_time'] = pd.to_datetime(
    train_df['pickup_time']).values.astype(float) / 10**9
test_df['pickup_time'] = pd.to_datetime(
    test_df['pickup_time']).values.astype(float) / 10**9

train_df['drop_time'] = pd.to_datetime(train_df['drop_time'])
test_df['drop_time'] = pd.to_datetime(test_df['drop_time'])

train_df['drop_time'] = pd.to_datetime(
    train_df['drop_time']).values.astype(float) / 10**9
test_df['drop_time'] = pd.to_datetime(
    test_df['drop_time']).values.astype(float) / 10**9

train_df['time_diff'] = train_df['drop_time'] - train_df['pickup_time']
test_df['time_diff'] = test_df['drop_time'] - test_df['pickup_time']

cols = list(train_df.columns)
cols.remove('label')
cols.remove('pick_lat')
cols.remove('pick_lon')
cols.remove('drop_lat')
cols.remove('drop_lon')
cols.remove('time_diff')
cols.remove('tripid')

for col in cols:
    train_df[col] = (train_df[col] - train_df[col].mean())/train_df[col].std()
    test_df[col] = (test_df[col] - test_df[col].mean())/test_df[col].std()

train_df = train_df.dropna()

print(train_df)


############


#######################################
feature_columns = []

time_as_a_numeric_column = tf.feature_column.numeric_column("time_diff")

time_boundaries = list(np.arange(int(min(train_df['time_diff'])),
                                 int(max(train_df['time_diff'])),
                                 60))

time_bucket = tf.feature_column.bucketized_column(time_as_a_numeric_column,
                                                  time_boundaries)

feature_columns.append(time_bucket)


resolution_in_degrees = 0.01
pic_latitude_as_a_numeric_column = tf.feature_column.numeric_column("pick_lat")

pic_latitude_boundaries = list(np.arange(int(min(train_df['pick_lat'])),
                                         int(max(train_df['pick_lat'])),
                                         resolution_in_degrees))

pic_latitude = tf.feature_column.bucketized_column(pic_latitude_as_a_numeric_column,
                                                   pic_latitude_boundaries)

# feature_columns.append(latitude)

# Create a bucket feature column for longitude.
pic_longitude_as_a_numeric_column = tf.feature_column.numeric_column(
    "pick_lon")
pic_longitude_boundaries = list(np.arange(int(min(train_df['pick_lon'])),
                                          int(max(train_df['pick_lon'])),
                                          resolution_in_degrees))

pic_longitude = tf.feature_column.bucketized_column(pic_longitude_as_a_numeric_column,
                                                    pic_longitude_boundaries)
# feature_columns.append(longitude)

# Create a feature cross of latitude and longitude.
pic_latitude_x_pic_longitude = tf.feature_column.crossed_column(
    [pic_latitude, pic_longitude], hash_bucket_size=100)
pic_crossed_feature = tf.feature_column.indicator_column(
    pic_latitude_x_pic_longitude)
feature_columns.append(pic_crossed_feature)

# drop

drop_latitude_as_a_numeric_column = tf.feature_column.numeric_column(
    "drop_lat")

drop_latitude_boundaries = list(np.arange(int(min(train_df['drop_lat'])),
                                          int(max(train_df['drop_lat'])),
                                          resolution_in_degrees))

drop_latitude = tf.feature_column.bucketized_column(drop_latitude_as_a_numeric_column,
                                                    drop_latitude_boundaries)

# feature_columns.append(latitude)

# Create a bucket feature column for longitude.
drop_longitude_as_a_numeric_column = tf.feature_column.numeric_column(
    "drop_lon")
drop_longitude_boundaries = list(np.arange(int(min(train_df['drop_lon'])),
                                           int(max(train_df['drop_lon'])),
                                           resolution_in_degrees))

drop_longitude = tf.feature_column.bucketized_column(drop_longitude_as_a_numeric_column,
                                                     drop_longitude_boundaries)
# feature_columns.append(longitude)

# Create a feature cross of latitude and longitude.
drop_latitude_x_drop_longitude = tf.feature_column.crossed_column(
    [drop_latitude, drop_longitude], hash_bucket_size=100)
drop_crossed_feature = tf.feature_column.indicator_column(
    drop_latitude_x_drop_longitude)
feature_columns.append(drop_crossed_feature)

# print(test_df_norm.head())

additional_fare = tf.feature_column.numeric_column("additional_fare")
feature_columns.append(additional_fare)

duration = tf.feature_column.numeric_column("duration")
feature_columns.append(duration)

meter_waiting = tf.feature_column.numeric_column("meter_waiting")
feature_columns.append(meter_waiting)

meter_waiting_fare = tf.feature_column.numeric_column("meter_waiting_fare")
feature_columns.append(meter_waiting_fare)

meter_waiting_till_pickup = tf.feature_column.numeric_column(
    "meter_waiting_till_pickup")
feature_columns.append(meter_waiting_till_pickup)

fare = tf.feature_column.numeric_column("fare")
feature_columns.append(fare)

time_diff = tf.feature_column.numeric_column("time_diff")
feature_columns.append(time_diff)

pickup_time = tf.feature_column.numeric_column("pickup_time")
feature_columns.append(pickup_time)

drop_time = tf.feature_column.numeric_column("drop_time")
feature_columns.append(drop_time)

feature_layer = layers.DenseFeatures(feature_columns)


def create_model(my_learning_rate, feature_layer, my_metrics):

    model = tf.keras.models.Sequential()

    model.add(feature_layer)

    # model.add(tf.keras.layers.Dense(
    #     units=10, activation=tf.keras.activations.relu))

    # model.add(tf.keras.layers.Dense(
    #     units=5, activation=tf.keras.activations.relu))

    model.add(tf.keras.layers.Dense(
        units=1, input_shape=(1,), activation=tf.keras.activations.sigmoid),)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(), metrics=my_metrics)

    return model


def train_model(model, dataset, epochs, label_name, batch_size=None, shuffle=True):

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label,
                        batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    epochs = history.epoch

    hist = pd.DataFrame(history.history)

    metrics = model.metrics

    return epochs, hist


def plot_curve(epochs, hist, list_of_metrics):

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        print(m)
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()


learning_rate = 0.001
epochs = 2000
batch_size = 2048
label_name = "label"
classification_threshold = 0.80

# Establish the metrics the model will measure.
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                    threshold=classification_threshold),
    tf.keras.metrics.Precision(thresholds=classification_threshold,
                               name='precision'
                               ),
    tf.keras.metrics.Recall(thresholds=classification_threshold,
                            name="recall"),
    tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', summation_method='interpolation', name="auc",
        dtype=None, thresholds=None, multi_label=False, label_weights=None
    )
]


my_model = create_model(learning_rate, feature_layer, METRICS)

# Train the model on the training set.
epoch, hist = train_model(my_model, train_df,
                          epochs, label_name, batch_size)


# Plot a graph of the metric(s) vs. epochs.
list_of_metrics_to_plot = ['accuracy', 'precision', "auc"]

# plot_curve(epoch, hist, list_of_metrics_to_plot)


def predict(n, label):

    fts = {name: np.array(value) for name, value in train_df.items()}

    predicted_values = my_model.predict_on_batch(x=fts)

    print("label          predicted")
    print("--------------------------------------")
    for i in range(n):
        # print("%5.0f %6.0f %15.0f" % (train_df[feature][10000 + i],
        #                               train_df[label][10000 + i],
        #                               predicted_values[i][0]))

        print(
            str(threshold((predicted_values[i][0]))) + "\t" + str(predicted_values[i][0]) + "\t" + str(train_df["label"][10000 + i]))


def threshold(n):

    if(n >= classification_threshold):
        return 1
    else:
        return 0


# predict(10, "label")


def predictTest():

    fts = {name: np.array(value) for name, value in test_df.items()}

    predicted_values = my_model.predict(x=fts)

    df = pd.DataFrame(columns=['tripid', 'prediction'])

    for i in range(8576):
        new_row = {'tripid': str(test_df["tripid"][i]), 'prediction': str(
            threshold((predicted_values[i][0])))}
        df = df.append(new_row, ignore_index=True)

    df.to_csv("./data/results.csv", index=False)


predictTest()
