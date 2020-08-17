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
start_future = dt.datetime(2020, 1, 1)
end_future = dt.datetime(2020, 8, 7)
base_version = ''

# Stock Info

stock_name = 'MSFT'
experiment_code = '0808_1739'
if stock_name == '^GSPC':
    base_version = '0727_1413'
if stock_name == '^DJI':
    base_version = '0728_1722'
if stock_name == '^IXIC':
    base_version = '0803_1132'
if stock_name == '^RUT':
    base_version = '0803_1633'
if stock_name == 'AAPL':
    base_version = '0807_1216'
if stock_name == 'MSFT':
    base_version = '0808_1219'
exp_name = 'up_fix_output'

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
try:
    trained_models.remove('Subprime')
    print('Loaded models: \n', *trained_models, sep = '\n')
except:
    print('Loaded models: \n', *trained_models, sep = '\n')

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
    y_pred_future = model.predict(x_future.reshape(x_future.shape[0], 45, 43, x_future.shape[-1], 1))
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

# Filter

best_balance = scores[scores['balance'] > 0.8]
print(best_balance)
best_models = best_balance[best_balance['trend'] > 0.5]
print(best_models)
# best_models_overall = best_models[best_models['mse']<np.mean(best_balance['mse'])]
# best_models_overall_trend = best_models[best_models['trend_acc'] > 1.9]
best_models_overall_trend = scores
print(best_models_overall_trend)
# Best model predictions

predictions_best_models = pd.DataFrame(columns=best_models_overall_trend.index)
for best_fit_models in best_models_overall_trend.index:
    model_to_load = dir_model+best_fit_models
    model = models.load_model(model_to_load, custom_objects={'trend_loss': trend_loss}, compile=True)
    y_pred_future = model.predict(x_future[-1].reshape(1, 45, 43, x_future.shape[-1], 1))
    y_desnorm_pred_future = finan.vect_desnorm(y, y_pred_future)
    close_pred = finan.close_converter(close_values, y_desnorm_pred_future[0])
#     predictions_best_models[best_fit_models] = y_desnorm_pred_future[0]
    predictions_best_models[best_fit_models] = close_pred
    print(best_fit_models)
    print(close_pred)

# y_pred_future = model.predict(x_future.reshape(x_future.shape[0], 45, 43, 26, 1))
mean_predictions = predictions_best_models.mean(axis=1)
std_predictions = predictions_best_models.mean(axis=1)

x_future_plot = np.arange(0, 5, 1)
x_past_plot = np.arange(-14, 1)

finan.dirmaker('Futurator/'+stock_name)

pred_close_dir = "Futurator/"+stock_name+"/y_pred_close_"+experiment_code+"_"+exp_name+".pkl"
f = open(pred_close_dir, "wb")
pickle.dump(predictions_best_models, f)
f.close()
print(close_values[-1])

plt.figure(figsize=(8, 18))
plt.plot(predictions_best_models)
plt.plot(x_future_plot, mean_predictions, color='red', linewidth=5, marker='o', markersize=10)
plt.plot(x_past_plot, close_values[-16:-1])
plt.legend(predictions_best_models)
plt.grid()
plt.xticks([0, 1, 2, 3, 4])
plt.xlim(0, 4)
plt.show()
