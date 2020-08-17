import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import finananal as finan
import datetime as dt
import tensorflow as tf
import pickle
import os
import yfinance as yf

from tensorflow.keras import models
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split

# Tensorboard & Tensorflow prep.

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Predictions dates

start = dt.datetime(1989, 1, 1)
end = dt.datetime(2020, 1, 9)
start_future = dt.datetime(2020, 1, 1)
end_future = dt.datetime(2020, 8, 7)
base_version = ''

# Training onditions

past_history = 45
future_target = 5
STEP = 1

# Stock Info

stock_name = '^GSPC'
experiment_code = '0813_1408'
version = "{}".format(dt.datetime.now().strftime("%m%d_%H%M"))
# if stock_name == '^GSPC':
#     base_version = '0727_1413'
# if stock_name == '^DJI':
#     base_version = '0728_1722'
# if stock_name == '^IXIC':
#     base_version = '0803_1132'
# if stock_name == '^RUT':
#     base_version = '0803_1633'
# if stock_name == 'AAPL':
#     base_version = '0807_1216'
# if stock_name == 'MSFT':
#     base_version = '0808_1219'
exp_name = 'up_fix_output'

# File Names

X_future_name = 'X_' + experiment_code + '.pkl'
y_future_name = 'y_' + experiment_code + '.pkl'
# X_future2pred_name = 'X_future2pred_' + experiment_code + '.pkl'
# X_close_future_name = 'X_future_close_' + experiment_code + '.pkl'
# dates_name = 'dates_' + experiment_code + '.pkl'

# Reading pickle
x_future = np.array(pd.read_pickle(r'normalized/' + stock_name + '/' + X_future_name))
y_future = np.array(pd.read_pickle(r'normalized/' + stock_name + '/' + y_future_name))
# x_predict = np.array(pd.read_pickle(r'data_train_val/' + stock_name + '/' + X_future2pred_name))
# x_close_future = np.array(pd.read_pickle(r'Future/' + stock_name + '/' + X_close_future_name))
# close_date_values = pd.read_pickle(r'Future/' + stock_name + '/' + dates_name)
# y = pd.read_pickle(r'multiplexed/' + stock_name + '/y_'+base_version+'.pkl')

X_data, y_data = finan.multivariate_data(dataset=x_future, target=y_future, start_index=0,
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

print('Single window of past history : {}'.format(X_train[0].shape))

BATCH_SIZE = 32
BUFFER_SIZE = 500

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_1last, y_train))
# train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_data_single = train_data_single.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_1last, y_val))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# Model dir

dir_model = 'saved_model/' + stock_name + '/' + exp_name + '/'
trained_models = os.listdir(dir_model)

try:
    trained_models.remove('Subprime')
    print('Loaded models: \n', *trained_models, sep = '\n')
except:
    print('Loaded models: \n', *trained_models, sep = '\n')

def tf_diff(a):
    return a[1:]-a[:-1]

def trend_loss(y_true, y_pred):
    y_true_sign = tf_diff(y_true)
    y_pred_sign = tf_diff(y_pred)
    trend_accuracy_penalty = tf.reduce_mean(tf.cast(tf.math.sign(y_true_sign) != tf.math.sign(y_pred_sign), tf.float32))
    loss = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    return loss*(trend_accuracy_penalty+1)

for model_to_analize in trained_models:
    print('Model:', model_to_analize)
    NAME = "{}".format(model_to_analize)
    model_to_load = dir_model+model_to_analize
    model = models.load_model(model_to_load, custom_objects={'trend_loss': trend_loss}, compile=True)

    log_dir = 'logs/fit/' + stock_name + '/' + exp_name + '/update_' + version + '/'+ model_to_analize
    print(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print(train_data_single)

    model_history = model.fit(train_data_single, epochs=20,
                                              steps_per_epoch=200,
                                              validation_data=val_data_single,
                                              validation_steps=20,
                                              callbacks=[tensorboard_callback])
    model.save('saved_model/' + stock_name + '/' + exp_name + '/update_' + version + '/' + NAME)