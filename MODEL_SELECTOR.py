import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import finananal as finan
import datetime as dt

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import metrics

import pickle

import os
import yfinance as yf

# Tensorboard & Tensorflow prep.

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Predictions dates

start = dt.datetime(1989, 1, 1)
end = dt.datetime(2020, 1, 9)
start_future = dt.datetime(2019, 7, 1)
end_future = dt.datetime(2020, 7, 29)
base_version = ''
# Stock Info

stock_name = '^GSPC'
experiment_code = '0730_1001'
if stock_name == '^GSPC':
    base_version = '0727_1413'
if stock_name == '^DJI':
    base_version = '0728_1722'
exp_name = 'up_normal'

# File Names

X_future_name = 'X_future_' + experiment_code + '.pkl'
y_future_name = 'y_future_' + experiment_code + '.pkl'
X_future2pred_name = 'X_future2pred_' + experiment_code + '.pkl'
X_close_future_name = 'X_future_close_' + experiment_code + '.pkl'
dates_name = 'dates_' + experiment_code + '.pkl'

# Reading pickle
x_future = np.array(pd.read_pickle(r'data_train_val/' + stock_name + '/' + X_future_name))
y_future = np.array(pd.read_pickle(r'data_train_val/' + stock_name + '/' + y_future_name))
x_predict = np.array(pd.read_pickle(r'data_train_val/' + stock_name + '/' + X_future2pred_name))
x_close_future = np.array(pd.read_pickle(r'Future/' + stock_name + '/' + X_close_future_name))
close_date_values = pd.read_pickle(r'Future/' + stock_name + '/' + dates_name)
y = pd.read_pickle(r'multiplexed/' + stock_name + '/y_'+base_version+'.pkl')

# Model dir

dir_model = 'saved_model/' + stock_name + '/' + exp_name + '/'
trained_models = os.listdir(dir_model)

# Future data

close_values = close_date_values.values
dates = close_date_values.index

# Future of the stock

stock_tick = yf.Ticker(stock_name)
index_trade = stock_tick.history(period="1y")
index_trade.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
index_trade = index_trade[start_future:end_future]

# Desnormalization

x_desnorm_future_undiff = np.array(index_trade['Close'])
y_desnorm_future = finan.vect_desnorm(y, y_future)
close_real = finan.close_converter(x_desnorm_future_undiff, y_desnorm_future[0])

# Model dataframes for performance creation

model_performance = pd.DataFrame(index=trained_models, columns=np.arange(0, y_future.shape[0], 1))
columns = ['mse', 'trend', 'pos_trend', 'neg_trend']
scores = pd.DataFrame(index=trained_models, columns=columns)
models_trend_acc = pd.DataFrame(columns=trained_models)
accurate_models = []

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
    model_to_load = dir_model+model_to_analize
    model = models.load_model(model_to_load, custom_objects={'trend_loss': trend_loss}, compile=True)
    y_pred_future = model.predict(x_future.reshape(x_future.shape[0], 45, 43, 39, 1))
    diff_pred_future = np.diff(y_pred_future)
    diff_future = np.diff(y_future)
    models_trend_acc[model_to_analize] = np.sum((np.sign(diff_future) == np.sign(diff_pred_future)), axis=1)
    y_desnorm_pred_future = finan.vect_desnorm(y, y_pred_future)
    close_pred = finan.close_converter(x_desnorm_future_undiff, y_desnorm_pred_future[0])
    acc_pos = finan.trend_acc(x_desnorm_future_undiff, y_desnorm_future, y_desnorm_pred_future, 1)
    acc_neg = finan.trend_acc(x_desnorm_future_undiff, y_desnorm_future, y_desnorm_pred_future, -1)
    model_mse = metrics.mse(y_desnorm_future, y_desnorm_pred_future)
    model_performance.loc[model_to_analize] = np.array(model_mse)
    print(np.average(model_mse))
    scores.loc[model_to_analize, 'mse'] = np.average(model_mse)
    scores.loc[model_to_analize, 'trend'] = acc_pos[0]
    scores.loc[model_to_analize, 'pos_trend'] = acc_pos[1]
    scores.loc[model_to_analize, 'neg_trend'] = acc_neg[1]
    if acc_pos[0] > 0.3:
        accurate_models.append(model_to_analize)
#         print(model.summary())
#         finan.multi_step_plot(x_desnorm_future_undiff[0][0], np.array(close_real), np.array(close_pred), 103)
        print('accuracy total:', acc_pos[0])
#         print('positive trend acc:', acc_pos[1])
#         print('negative trend acc:', acc_neg[1])
    else:
        print('Model not accurate enough')
scores['balance'] = (scores['pos_trend']*scores['neg_trend'])/(scores['pos_trend']+scores['neg_trend'])*4

# model performace tranpose

model_performance = model_performance.T.sort_index(axis=0)

# Trend acc

scores['trend_acc'] = models_trend_acc.sum()/len(models_trend_acc)

print(scores)