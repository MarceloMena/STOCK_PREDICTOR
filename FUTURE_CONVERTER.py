import numpy as np
import pandas as pd
import finananal as finan
import matplotlib.pyplot as plt

import datetime as dt

import pickle
import yfinance as yf

# Stock query

stock_name = 'MSFT'
version = "{}".format(dt.datetime.now().strftime("%m%d_%H%M"))
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

stock_tick = yf.Ticker(stock_name)
stock = stock_tick.history(period="2y")
stock.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
stock['Close Change'] = stock.pct_change()['Close']

# Loading fundamental analysis

# stock_fun_an = pd.read_csv('Future/STOCK_FUN_AN', parse_dates=True, index_col=0)
stock_fun_an = []
print('fundamental analysis loaded')
# Time Period

start = dt.datetime(2019, 6, 1)
end = dt.datetime(2020, 8, 4)
real_close_values = stock['Close'][start:end]
dates = stock[start:end]
stock = stock[start:end]

# Generation of the data

all_features, all_y = finan.generator(stock, stock_fun_an, start=start, end=end, days_an=45)
frames = finan.multiplexor(all_features)
print('Generating data')
# Reading desnormalization files

stock_past = pd.read_pickle(r'multiplexed/'+stock_name+'/X_'+base_version+'.pkl')
y_past = pd.read_pickle(r'multiplexed/'+stock_name+'/y_'+base_version+'.pkl')

# Normalization

stock_future_norm = finan.full_data_norm_future(frames, stock_past)

array_data_stock = []
for data in stock_future_norm:
    array_data_stock.append(data.values)
array_data_stock = np.array(array_data_stock)

y_norm_future = (all_y - y_past.min()) / (y_past.max() - y_past.min())
y_norm_future = np.array(y_norm_future)
print('Normalizing data')
# Data Structure

TRAIN_SPLIT = 5
past_history = 45
future_target = 5
STEP = 1


def multivariate_data_dates(dataset, start_index, history_size,
                            step, single_step=False):

    data = []
    labels = []

    start_index = start_index + history_size
    end_index = len(dataset)

    for i in range(start_index, end_index):
        first_ind = i - history_size
        end_ind = i
        #         print(dataset[first_ind:end_ind])
        data.append(dataset[first_ind:end_ind])

    return np.array(data), dataset.index[start_index:end_index]


x_real_close, dates_future = multivariate_data_dates(dataset=real_close_values, start_index=0,
                                                     history_size=past_history,
                                                     step=STEP,
                                                     single_step=False)

x_train_future, y_train_future = finan.multivariate_data(dataset=array_data_stock, target=y_norm_future, start_index=0,
                                                         end_index=None, history_size=past_history,
                                                         target_size=future_target, step=STEP,
                                                         single_step=False)

x_future_to_predict, y = finan.multivariate_data(dataset=array_data_stock, target=y_norm_future, start_index=0,
                                                 end_index=None, history_size=past_history,
                                                 target_size=0, step=STEP,
                                                 single_step=False)
print('Saving data version:', version)

finan.dirmaker('data_train_val/'+stock_name)

data_train_val_dir_X = "data_train_val/"+stock_name+"/X_future_"+version+".pkl"
f = open(data_train_val_dir_X, "wb")
pickle.dump(x_train_future, f)
f.close()

data_train_val_dir_y = "data_train_val/"+stock_name+"/y_future_"+version+".pkl"
f = open(data_train_val_dir_y, "wb")
pickle.dump(y_train_future, f)
f.close()

data_train_val_dir_X_future = "data_train_val/"+stock_name+"/X_future2pred_"+version+".pkl"
f = open(data_train_val_dir_X_future, "wb")
pickle.dump(x_future_to_predict, f)
f.close()

finan.dirmaker('Future/'+stock_name)

future_close = "Future/"+stock_name+"/X_future_close_"+version+".pkl"
f = open(future_close, "wb")
pickle.dump(x_real_close, f)
f.close()

dates_close = "Future/"+stock_name+"/dates_"+version+".pkl"
f = open(dates_close, "wb")
pickle.dump(real_close_values, f)
f.close()

plt.figure(figsize=(12, 12))
plt.imshow(array_data_stock[0])
plt.title(dates.index[0], fontsize=20)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Days of techinal analisys', fontsize=16)
plt.colorbar()
plt.show()

plt.figure(figsize=(12, 12))
plt.imshow(array_data_stock[-1])
plt.title(dates.index[-1], fontsize=20)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Days of techinal analisys', fontsize=16)
plt.colorbar()
plt.show()
