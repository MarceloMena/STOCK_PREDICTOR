import tensorflow as tf
import datetime

import numpy as np
import pandas as pd
import finananal as finan

from tensorflow.keras import layers
from tensorflow.keras import models

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.optimizers import SGD

sgd = SGD(lr=0.05)

# Tensorboard & Tensorflow prep.

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Split seq2seq

# TRAIN_SPLIT = 7000
past_history = 45
future_target = 1
STEP = 1

stock_name = '^GSPC'
if stock_name == '^GSPC':
    experiment_code = '0810_1721'
if stock_name == '^DJI':
    experiment_code = '0728_1722'
if stock_name == '^IXIC':
    experiment_code = '0803_1132'
if stock_name == '^RUT':
    experiment_code = '0803_1633'
if stock_name == 'AAPL':
    experiment_code = '0807_1551'
if stock_name == 'MSFT':
    experiment_code = '0808_1219'
# normalized folder
exp_name = 'one_day_normal_output_layer'

# Reading the data

stock_data = pd.read_pickle(r'normalized/' + stock_name + '/X_' + experiment_code + '.pkl')
stock_data = np.array(stock_data, dtype='f')
stock_data = stock_data[1:]

labels = pd.read_pickle(r'normalized/' + stock_name + '/labels_' + experiment_code + '.pkl')

# One hot encoding

data = labels
values = np.array(data)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# Creating the seqs

X_data, y_data = finan.multivariate_data(dataset=stock_data, target=onehot_encoded, start_index=0,
                                         end_index=None, history_size=past_history,
                                         target_size=future_target, step=STEP,
                                         single_step=False)

# Train-val split

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2)

# Params extractions for training

NUM_FEAT = X_val.shape[-1]
cubes_val = X_val.shape[0]
cubes_train = X_train.shape[0]
x_train_1last = X_train.reshape(cubes_train, past_history, X_train.shape[2], NUM_FEAT, 1)
x_val_1last = X_val.reshape(cubes_val, past_history, X_val.shape[2], NUM_FEAT, 1)

num_cat = y_train.shape[-1]
ohe_train = y_train.shape[0]
ohe_val = y_val.shape[0]
y_train_1last = y_train.reshape(ohe_train, num_cat)
y_val_1last = y_val.reshape(ohe_val, num_cat)

print('Single window of past history : {}'.format(X_train[0].shape))

BATCH_SIZE = 32
BUFFER_SIZE = 500

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_1last, y_train_1last))
# train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_data_single = train_data_single.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_1last, y_val_1last))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


# def tf_diff(a):
#     return a[1:] - a[:-1]
#
#
# def trend_loss(y_true, y_pred):
#     y_true_sign = tf_diff(y_true)
#     y_pred_sign = tf_diff(y_pred)
#     trend_accuracy_penalty = tf.reduce_mean(tf.cast(tf.math.sign(y_true_sign) != tf.math.sign(y_pred_sign), tf.float32))
#     loss = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
#     return loss * (trend_accuracy_penalty + 1)


conv_layers = [2]
start_filters = [64]
lstm_layers = [2]
dense_layers = [3]
print(train_data_single)
for conv_layer in conv_layers:
    for start_filter in start_filters:
        for lstm_layer in lstm_layers:
            for dense_layer in dense_layers:
                NAME = "{}-Conv-{}-N-{}-R-{}-D-{}".format(conv_layer, start_filter, lstm_layer, dense_layer,
                                                          datetime.datetime.now().strftime("%m%d-%H%M"))
                print(NAME)
                multi_step_model = models.Sequential()

                multi_step_model.add(
                    layers.InputLayer(batch_input_shape=(None, past_history, X_train.shape[2], NUM_FEAT, 1)))
                multi_step_model.add(
                    layers.TimeDistributed(layers.Conv2D(start_filter, (5, 5), activation='relu'), name='Conv1'))
                multi_step_model.add(layers.TimeDistributed(layers.MaxPooling2D(2, 2), name='Pooling1'))
                multi_step_model.add(tf.keras.layers.TimeDistributed(layers.Dropout(0.25), name='Drop1'))

                if conv_layer >= 2:
                    multi_step_model.add(
                        layers.TimeDistributed(layers.Conv2D(int(start_filter * 2), (3, 3), activation='relu'),
                                               name='Conv2'))
                    multi_step_model.add(layers.TimeDistributed(layers.MaxPooling2D(2, 2), name='Pooling2'))
                    multi_step_model.add(tf.keras.layers.TimeDistributed(layers.Dropout(0.25), name='Drop2'))

                if conv_layer >= 3:
                    multi_step_model.add(
                        layers.TimeDistributed(layers.Conv2D(int(start_filter * 4), (3, 3), activation='relu'),
                                               name='Conv3'))
                    multi_step_model.add(layers.TimeDistributed(layers.MaxPooling2D(2, 2), name='Pooling3'))
                    multi_step_model.add(tf.keras.layers.TimeDistributed(layers.Dropout(0.25), name='Drop3'))

                multi_step_model.add(layers.TimeDistributed(layers.Flatten(), name='Flat'))

                if lstm_layer == 1:
                    multi_step_model.add(layers.GRU(256, name='GRU1'))

                if lstm_layer >= 2:
                    multi_step_model.add(layers.GRU(256, return_sequences=True, name='GRU1'))
                    multi_step_model.add(layers.GRU(128, name='GRU2'))

                if dense_layer >= 3:
                    multi_step_model.add(layers.Dense(128))
                if dense_layer >= 2:
                    multi_step_model.add(layers.Dense(64))

                multi_step_model.add(layers.Dense(y_train.shape[-1], activation='sigmoid'))

                multi_step_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

                print(multi_step_model.summary())
                multi_step_model.save('saved_model/' + stock_name + '/' + exp_name + '/' + NAME)

                log_dir = 'logs/fit/' + stock_name + '/' + exp_name + '/' + NAME
                print(log_dir)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                print(train_data_single)

                multi_step_history = multi_step_model.fit(train_data_single, epochs=50,
                                                          steps_per_epoch=500,
                                                          validation_data=val_data_single,
                                                          validation_steps=100,
                                                          callbacks=[tensorboard_callback])
                multi_step_model.save('saved_model/' + stock_name + '/' + exp_name + '/' + NAME)
