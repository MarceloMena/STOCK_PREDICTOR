import numpy as np
import pandas as pd
import finananal as finan
import datetime as dt
import matplotlib.pyplot as plt
import pickle

import os

# Stock information

stock_name = '^GSPC'
version = "{}".format(dt.datetime.now().strftime("%m%d_%H%M"))

stock_dir = '/home/marcelo/Downloads/'+stock_name+'.csv'

index_trade = pd.read_csv(stock_dir, parse_dates=True, index_col=0)
index_trade = index_trade.asfreq(freq='B', method = 'pad')
index_trade = index_trade.drop('Adj Close', axis =1)
index_trade['Close Change'] = np.sign(index_trade.pct_change()['Close'])
print('loaded historical data')

index_trade.drop(index_trade.index[index_trade['Close'].isnull()], inplace=True)

# fundamental analysis information

# fun_anal_dir = '/home/marcelo/Documents/IA/SP500_FUN_AN'

# stock_fun_an = pd.read_csv('Future/STOCK_FUN_AN_062020', parse_dates=True, index_col=0)

index_trade_fun_an_past = pd.read_csv('/home/marcelo/Documents/IA/SP500_FUN_AN', parse_dates=True, index_col=0)
index_trade_fun_an_past = index_trade_fun_an_past.loc[:'2019-12-31']

index_trade_fun_an = pd.read_csv('Future/STOCK_FUN_AN', parse_dates=True, index_col=0)

updates_index_trade_fun_an = pd.concat([index_trade_fun_an_past.loc[:'2019-12-31'],
                                        index_trade_fun_an.loc['2020-01-01':]])

print('Fundamental analysis')

# Train dates

start = dt.datetime(1990, 1, 1)
end = dt.datetime(2020, 6, 1)
# start_plot = dt.datetime(2019, 6, 1)
# end_plot = dt.datetime(2020, 1, 9)

# Data generation

all_features, all_y = finan.generator(tec_an=index_trade,
                                      fun_an= updates_index_trade_fun_an,
                                      start = start,
                                      end = end,
                                      days_an = 45)

y_index_trade = all_y.drop(all_y.index[0])

movement = np.sign(all_y)
cut_movement = movement[1:]
labels = []
for i in cut_movement:
    if i == -1:
        labels.append(0)
    else:
        labels.append(1)

finan.dirmaker('multiplexed/'+stock_name)

multiplexed_dir_y = "multiplexed/"+stock_name+"/y_"+version+".pkl"
f = open(multiplexed_dir_y,"wb")
pickle.dump(labels,f)
f.close()

print('Data generated')

# Data Preparation

X_index_trade = finan.multiplexor(all_features)
multiplexed_dir_X = "multiplexed/"+stock_name+"/X_"+version+".pkl"
f = open(multiplexed_dir_X,"wb")
pickle.dump(X_index_trade,f)
f.close()
print('Data preparation')

# Normalization

# y_norm = (y_index_trade - y_index_trade.min()) / (y_index_trade.max() - y_index_trade.min())
# y_norm = y_index_trade
y_norm = (movement - movement.min()) / (movement.max() - movement.min())
X_norm = finan.full_data_norm(X_index_trade)

array_data_index_trade = []
for data in X_norm:
    assert isinstance(data.values, object)
    array_data_index_trade.append(data.values)
array_data_index_trade = np.array(array_data_index_trade)
print('Data Normalized')

# Save files

finan.dirmaker('normalized/'+stock_name)

normalized_dir_X = "normalized/"+stock_name+"/X_"+version+".pkl"
f = open(normalized_dir_X,"wb")
pickle.dump(array_data_index_trade,f)
f.close()

normalized_dir_y = "normalized/"+stock_name+"/labels_"+version+".pkl"
f = open(normalized_dir_y,"wb")
pickle.dump(labels,f)
f.close()

plt.figure(figsize=(12,12))
plt.imshow(X_norm[150])
plt.title(y_norm.index[110], fontsize =20)
plt.xlabel('Features', fontsize= 16)
plt.ylabel('Days of techinal analisys', fontsize= 16)
plt.colorbar()
plt.show()

print(version)