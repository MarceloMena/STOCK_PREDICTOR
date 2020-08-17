import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

stock_name = 'MSFT'
experiment_code = '0808_1739'
exp_name = 'up_fix_output'

# File Names

y_pred_close_dir = 'y_pred_close_' + experiment_code + '_'+exp_name + '.pkl'
dates_name = 'dates_' + experiment_code + '.pkl'

y_pred_close = np.array(pd.read_pickle(r'Futurator/'+stock_name+'/'+y_pred_close_dir))
close_date_values = pd.read_pickle(r'Future/' + stock_name + '/' + dates_name)

close_values = close_date_values.values
dates = close_date_values.index

w = y_pred_close
x = np.arange(0, 6, 1)
y = close_values[-16:]
z = np.arange(-15, 1, 1)

std_dev = np.std(w, axis=1)
mean_w = np.mean(w, axis=1)

mean_w = np.insert(mean_w, 0, y[-1])
std_dev = np.insert(std_dev, 0, 0)

print(close_date_values.values)
print(y)
print(std_dev)
print(mean_w)

tomorrow_date = dates[-1]
# last_date = dates[-1]+timedelta(wd=5)

plt.figure(figsize=(12, 6))
# plt.plot(x,w)
plt.title("{} prediction {}".format(stock_name, tomorrow_date.strftime("%m/%d")), fontsize=20)
plt.plot(z, y, color='g')
plt.plot(x, mean_w, color='orange', linewidth=2)
plt.fill_between(x, (mean_w-std_dev), (mean_w+std_dev), color='orange', alpha=.1)
plt.fill_between(x, (mean_w-std_dev/2), (mean_w+std_dev/2), color='orange', alpha=.2)
plt.xlim(-15, 5)
plt.xticks([-15, -10, -5, 0, 5])
plt.grid()
plt.savefig('Futurator/'+stock_name+'/'+experiment_code+'_'+exp_name+'.png')
plt.show()
