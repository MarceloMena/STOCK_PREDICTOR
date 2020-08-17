import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
import os

from datetime import datetime, timedelta

### Calculo de indicadores de analisis tecnico -  Analisis tecnico

def technical_analysis(data, period = 1, lbp_R = 1, pow1_KAMA = 2,
                       pow2_KAMA = 20,
                       fillna = True,
                       r_RSI = 25,
                       s_RSI = 13,
                       d_n_stock_osc = 3,
                       ndev_bb = 2):
    #Trends:

    ADXIndicator_SP_500 = ta.trend.ADXIndicator(data['High'],
                                                data['Low'],
                                                data['Close'],
                                                n=period)
    data['ADX'] = ADXIndicator_SP_500.adx()
    data['ADX neg'] = ADXIndicator_SP_500.adx_neg()
    data['ADX pos'] = ADXIndicator_SP_500.adx_pos()
    EMAIndicator_SP_500_5d = ta.trend.EMAIndicator(data['Close'],
                                                   n=period)
    data['EMA'] = EMAIndicator_SP_500_5d.ema_indicator()
    DPOIndicator_SP_500_5d = ta.trend.DPOIndicator(data['Close'],
                                                   n=period)
    data['DPO'] = DPOIndicator_SP_500_5d.dpo()

    #Momentum

    WR_SP_500 = ta.momentum.wr(data['High'],
                               data['Low'],
                               data['Close'],
                               lbp = lbp_R)
    data['Williams %R'] = WR_SP_500
    ROCIndicator_SP500 = ta.momentum.ROCIndicator(data['Close'],
                                                  n=period,
                                                  fillna=True)
    data['ROC'] = ROCIndicator_SP500.roc()
    KAMAIndicator_SP500 = ta.momentum.KAMAIndicator (data['Close'], n = period, pow1 = pow1_KAMA,
                                                     pow2 = pow2_KAMA,
                                                     fillna = True)
    data['KAMA'] = KAMAIndicator_SP500.kama()
    MFIIndicator_SP500 = ta.volume.money_flow_index(data['High'],
                                                  data['Low'],
                                                  data['Close'],
                                                  data['Volume'],
                                                  n=period,
                                                  fillna=True)
    data['MFI'] = MFIIndicator_SP500
    RSIIndicator_SP500 = ta.momentum.RSIIndicator(data['Close'],
                                                  n=period,
                                                  fillna=True)
    data['RSI'] = RSIIndicator_SP500.rsi()
    TSIIndicator_SP500 = ta.momentum.TSIIndicator(data['Close'],
                                                  r = r_RSI,
                                                  s = s_RSI,
                                                  fillna = True)
    data['TSI'] = TSIIndicator_SP500.tsi()

    #Oscillator

    StochasticOscillator_SP500 = ta.momentum.StochasticOscillator(
                                                                  data['High'],
                                                                  data['Low'], data['Close'],
                                                                  n = period,
                                                                  d_n = d_n_stock_osc,
                                                                  fillna = True)
    data['Stock Osc'] = StochasticOscillator_SP500.stoch()
    data['Stock Osc sig'] = StochasticOscillator_SP500.stoch_signal()

    #Volume

    AccDistIndexIndicator_SP500 = ta.volume.acc_dist_index(data['High'],
                                                           data['Low'],
                                                           data['Close'],
                                                           data['Volume'],
                                                           fillna=True)
    data['ADI'] = AccDistIndexIndicator_SP500
    ChaikinMoneyFlowIndicator_SP_500 = ta.volume.ChaikinMoneyFlowIndicator(data['High'],
                                                                           data['Low'],
                                                                           data['Close'],
                                                                           data['Volume'],
                                                                           n =period)
    data['Chainkin MF'] = ChaikinMoneyFlowIndicator_SP_500.chaikin_money_flow()
    EaseOfMovementIndicator_SP_500 = ta.volume.EaseOfMovementIndicator(data['High'],
                                                                       data['Low'],
                                                                       data['Volume'],
                                                                       n =period)
    data['EoM'] = EaseOfMovementIndicator_SP_500.ease_of_movement()
    data['EoM SMA'] = EaseOfMovementIndicator_SP_500.sma_ease_of_movement()

    #Volatility

    BollingerBands = ta.volatility.BollingerBands(data['Close'], n=period, ndev = ndev_bb)
    data['Bol Band mavg'] = BollingerBands.bollinger_mavg()
    data['Bol Band high'] = BollingerBands.bollinger_hband()
    data['Bol Band low'] = BollingerBands.bollinger_lband()
    AverageTrueRange_SP_500_5d = ta.volatility.AverageTrueRange(data['High'],
                                                                data['Low'],
                                                                data['Close'],
                                                                n = period)
    data['ATR'] = AverageTrueRange_SP_500_5d.average_true_range()
    DonchianChannel_SP_500_5d = ta.volatility.DonchianChannel(data['High'],
                                                              data['Low'],
                                                              data['Close'],
                                                              n=period)
    data['Donchian hi-band'] = DonchianChannel_SP_500_5d.donchian_channel_hband()
    data['Donchian hi-ind'] = DonchianChannel_SP_500_5d.donchian_channel_hband_indicator()
    data['Donchian lo-band'] = DonchianChannel_SP_500_5d.donchian_channel_lband()
    data['Donchian lo-ind'] = DonchianChannel_SP_500_5d.donchian_channel_lband_indicator()

    return data

### Plot de indicadores de Hi/lo/Close -  Analisis tecnico

def tec_an_HLC_plot(SP500_plot, start, end, days):

    SP500_plot = SP500_plot.loc[start:end]

    fig = plt.figure(figsize=(16,12))

    ax1 = plt.subplot2grid((12,1), (0,0), rowspan=1, colspan=1)
    ax1.set_ylabel('Williams %R')
    ax1.set_xticklabels([])

    ax1.plot(SP500_plot.index,
            SP500_plot['Williams %R'],
            color = 'magenta'
           )
    plt.grid()
    plt.xlim(start, end)
    ax2 = plt.subplot2grid((12,1), (1,0), rowspan=1, colspan=1)
    ax2.set_ylabel('ADX')
    ax2.set_xticklabels([])

    ax2.plot(SP500_plot.index,
            SP500_plot['ADX'],
            color = 'black'
           )
    plt.grid()
    plt.xlim(start, end)
    ax3 = plt.subplot2grid((12,1), (2,0), rowspan=1, colspan=1)
    ax3.set_ylabel('ADX neg')
    ax3.set_xticklabels([])

    ax3.plot(SP500_plot.index,
            SP500_plot['ADX neg'],
            color = 'red'
           )
    plt.grid()
    plt.xlim(start, end)
    ax4 = plt.subplot2grid((12,1), (3,0), rowspan=1, colspan=1)
    ax4.set_ylabel('ADX pos')
    ax4.set_xticklabels([])

    ax4.plot(SP500_plot.index,
            SP500_plot['ADX pos'],
            color = 'green'
           )
    plt.grid()
    plt.xlim(start, end)
    ax5 = plt.subplot2grid((12,1), (4,0), rowspan=6, colspan=1)
    ax5.set_ylabel('SP_500')
    ax5.set_xticklabels([])

    ax5.plot(SP500_plot.index,
             SP500_plot['Close'], alpha = 0.5, label = 'SP500'
             )
    ax5.plot(SP500_plot.index,
             SP500_plot[['Bol Band high', 'Bol Band low']],
             color = 'orange',
             alpha = 0.7, label = 'Bollinger Bands'
            )
    ax5.plot(SP500_plot.index,
             SP500_plot['Bol Band mavg'],
             color = 'green', label = 'Bollinger Bands mavg'
            )

    ax5.legend(loc = 2)

    plt.grid()
    plt.xlim(start, end)
    ax6 = plt.subplot2grid((12,1), (10,0), rowspan=1, colspan=1)
    ax6.set_ylabel('Stock Osc')
    ax6.set_xticklabels([])

    ax6.plot(SP500_plot.index, SP500_plot['Stock Osc'],
             color = 'k'
            )
    plt.grid()
    plt.xlim(start, end)
    ax7 = plt.subplot2grid((12,1), (11,0), rowspan=1, colspan=1)
    ax7.set_ylabel('ATR')

    ax7.plot(SP500_plot.index, SP500_plot['ATR'],
             color = 'k'
            )
    plt.grid()
    plt.xlim(start, end)
    plt.suptitle(f'Hi-Lo-Close Indexes for a period of {days} days \n from {start} to {end}', fontsize = 20)
    plt.savefig(f'/home/marcelo/Documents/IA/SP500_TEC_AN(HLC)_{days}days.png')
    plt.show()

### Plot de indicadores de Close -  Analisis tecnico

def tec_an_close_plot(data, start, end, days):

    data = data.loc[start:end]

    fig = plt.figure(figsize=(16,12))

    ax1 = plt.subplot2grid((12,1), (0,0), rowspan=1, colspan=1)
    ax1.set_ylabel('DPO')
    ax1.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)
    ax1.plot(data.index,
            data['DPO'],
            color = 'magenta'
           )

    ax2 = plt.subplot2grid((12,1), (1,0), rowspan=1, colspan=1)
    ax2.set_ylabel('ROC')
    ax2.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)
    ax2.plot(data.index,
            data['ROC'],
            color = 'black'
           )

    ax3 = plt.subplot2grid((12,1), (2,0), rowspan=1, colspan=1)
    ax3.set_ylabel('RSI')
    ax3.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax3.plot(data.index,
            data['RSI'],
            color = 'red'
           )

    ax4 = plt.subplot2grid((12,1), (3,0), rowspan=1, colspan=1)
    ax4.set_ylabel('TSI')
    ax4.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax4.plot(data.index,
            data['TSI'],
            color = 'green'
           )

    ax5 = plt.subplot2grid((12,1), (4,0), rowspan=6, colspan=1)
    ax5.set_ylabel('SP_500')
    ax5.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax5.plot(data.index,
             data['Close'], alpha = 0.5, label = 'SP500'
             )
    ax5.plot(data.index,
             data[['Donchian hi-band', 'Donchian lo-band']],
             color = 'red',
             alpha = 0.9, label = 'Donochian Bands'
            )
    ax5.legend(loc = 2)

    ax6 = plt.subplot2grid((12,1), (10,0), rowspan=1, colspan=1)
    ax6.set_ylabel('Donchian hi', fontsize = 8)
    ax6.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax6.plot(data.index, data['Donchian hi-ind'],
             color = 'k'
            )
    ax7 = plt.subplot2grid((12,1), (11,0), rowspan=1, colspan=1)
    ax7.set_ylabel('Donchian lo', fontsize = 8)
    plt.grid()
    plt.xlim(start, end)

    ax7.plot(data.index, data['Donchian lo-ind'],
             color = 'k'
            )
    plt.suptitle(f'Close Indexes for a period of {days} days \n from {start} to {end}', fontsize = 20)
    plt.savefig(f'/home/marcelo/Documents/IA/SP500_TEC_AN(Close)_{days}days.png')
    plt.show()

### Plot de indicadores de Volumen -  Analisis tecnico

def tec_an_volume_plot(data, start, end, days):

    data = data.loc[start:end]

    fig = plt.figure(figsize=(16,12))

    ax1 = plt.subplot2grid((12,1), (0,0), rowspan=1, colspan=1)
    ax1.set_ylabel('MFI')
    ax1.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax1.plot(data.index,
            data['MFI'],
            color = 'magenta'
           )

    ax2 = plt.subplot2grid((12,1), (1,0), rowspan=1, colspan=1)
    ax2.set_ylabel('ADI')
    ax2.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax2.plot(data.index,
            data['ADI'],
            color = 'black'
           )

    ax3 = plt.subplot2grid((12,1), (2,0), rowspan=1, colspan=1)
    ax3.set_ylabel('Chainkin MF')
    ax3.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax3.plot(data.index,
            data['Chainkin MF'],
            color = 'red'
           )

    ax4 = plt.subplot2grid((12,1), (3,0), rowspan=1, colspan=1)
    ax4.set_ylabel('TSI')
    ax4.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax4.plot(data.index,
            data['TSI'],
            color = 'green'
           )

    ax5 = plt.subplot2grid((12,1), (4,0), rowspan=6, colspan=1)
    ax5.set_ylabel('Volume 1x10⁹ USD')
    ax5.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax5.plot(data.index,
             data['Volume'],alpha = 0.7,color = 'green', label = 'Volume'
             )
    ax5.legend(loc = 2)
    ax15 = ax5.twinx()
    labels = ['EMA','KAMA']
    ax15.plot(data.index,
             data['KAMA'],
             color = 'orange', label = 'KAMA',
              alpha = 0.7
            )
    ax15.plot(data.index,
             data['EMA'],
             color = 'purple', label = 'EMA',
              alpha = 0.7
            )
    ax15.plot(data.index,
             data['Close'],alpha = 0.5, color = 'blue', label = 'SP 500 Close'
             )
    ax15.legend(loc = 2)
    ax15.set_xticklabels([])

    ax6 = plt.subplot2grid((12,1), (10,0), rowspan=1, colspan=1)
    ax6.set_ylabel('EoM')
    ax6.set_xticklabels([])
    plt.grid()
    plt.xlim(start, end)

    ax6.plot(data.index, data['EoM'],
             color = 'k'
            )

    ax7 = plt.subplot2grid((12,1), (11,0), rowspan=1, colspan=1)
    ax7.set_ylabel('EoM SMA')
    plt.grid()
    plt.xlim(start, end)

    ax7.plot(data.index, data['EoM SMA'],
             color = 'k'
            )
    plt.suptitle(f'Volume Indexes for a period of {days} days \n from {start} to {end}', fontsize = 20)
    plt.savefig(f'/home/marcelo/Documents/IA/SP500_TEC_AN(Volume)_{days}days.png')

    plt.show()

### Normalizador de dataframes

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

### Preparacion de datos

def data_sampler_norm(data, period, start, end, type_an='all'):
    if type_an == 'all':
        data = data
    if type_an == 'basic_tec':
        data = data.loc[:, 'Open':'Volume']
    if type_an == 'full_tec':
        data = data.loc[:, 'Open':'Donchian lo-ind']
    if type_an == 'no_sentiment':
        data = data.loc[:, 'Open':'DCOILWTICO']
    if type_an == 'full_fun':
        close = data.Close
        data = data.loc[:, 'GDPC1':'Bearish']
        data = data.join(close)
    data = data.loc[start:end]
    data_sampled = data.asfreq(freq=period)
    data_sampled_norm = normalize(data_sampled)
    y = data_sampled_norm.Close
    y = y.to_frame()
    X_sampled_norm = data_sampled_norm.drop('Close', axis = 1)
    return X_sampled_norm, y

def data_merger(df_tec_an, df_fun_an, start, end):
    df_tec_an = df_tec_an.loc[start:end]
    df_fun_an = df_fun_an.loc[start:end]
    df_fun_an = df_fun_an.drop('S&P 500 Weekly Close', axis = 1)
    sp500_fin_an = df_tec_an.join(df_fun_an)
    return sp500_fin_an

def generator(tec_an, fun_an, start, end, days_an = 90, dashboard = 'normal'):
    all_y = tec_an.loc[start:end]['Close Change']
    all_features = {2: days_an}
    true_start = start - timedelta(days=1)
    if len(fun_an) == 0:
        tec_only = True
    else:
        tec_only = False
    for i in range(2,days_an,1):
        period = i
        # print(period)
        sp500_tec_an = technical_analysis(tec_an, period = period, lbp_R = period, d_n_stock_osc=3)
        sp500_tec_an = sp500_tec_an.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
        if tec_only == False:
            sp500_fin_an = data_merger(sp500_tec_an, fun_an, start = start, end = end)
        else:
            sp500_fin_an = sp500_tec_an.loc[start:end]
        if dashboard == 'diff':
            sp500_fin_an = sp500_fin_an.diff().drop(sp500_fin_an.index[0])
        if dashboard == 'pct':
            sp500_fin_an = sp500_fin_an.pct_change().drop(sp500_fin_an.index[0])
        all_features[i] = sp500_fin_an
    return all_features, all_y

def multiplexor(df):
    frames = []
    fix_period_data = df[2]
    for date in range(len(df[2].index)):
        reciever = pd.DataFrame(columns = fix_period_data.columns)
        for days in df.keys():
            fix_period_data = df[days]
            reciever = reciever.append(fix_period_data.iloc[date], ignore_index = True)
        frames.append(reciever)
    return frames

def full_data_norm(df):
    result = df.copy()
    super_min =[]
    super_max =[]
    array_data = []
    for feature_name in result[1].columns:
        minimum = []
        maximum = []
        for i in range(0,len(df),1):
            minimum.append(df[i][feature_name].min())
            maximum.append(df[i][feature_name].max())
        super_min=(np.array(minimum).min())
        super_max=(np.array(maximum).max())
        for i in range(0,len(df),1):
            result[i][feature_name] = (df[i][feature_name] - super_min) / (super_max - super_min)
    return result

def flattener(df):
    df_flattened=[]
    for i in range(df.shape[0]):
        df_flattened.append(df[i].flatten())
    return np.array(df_flattened)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        # print(end_index)

    for i in range(start_index, end_index-1):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i+1:i+target_size+1])

    return np.array(data), np.array(labels)

### Normalizador de dataframes

def full_data_norm_future(df, df_past):
    result = df.copy()
    super_min =[]
    super_max =[]
    array_data = []
    for feature_name in result[1].columns:
        minimum = []
        maximum = []
        for i in range(0,len(df_past),1):
            minimum.append(df_past[i][feature_name].min())
            maximum.append(df_past[i][feature_name].max())
        super_min=(np.array(minimum).min())
        super_max=(np.array(maximum).max())
        for i in range(0,len(df),1):
            result[i][feature_name] = (df[i][feature_name] - super_min) / (super_max - super_min)
    return result

### Time steps para el pasado

def create_time_steps(length):
    return list(range(-length, 0))

### plot multipasos

def multi_step_plot(history, true_future, prediction,day):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.title(day, fontsize =20)
    plt.plot(num_in, np.array(history), label='History')
#     print(np.array(history[:, 1]))
    plt.plot(np.arange(num_out), np.array(true_future), 'b',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'r',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('Days', fontsize= 16)
    plt.ylabel('SP500 value', fontsize= 16)
    plt.show()

## Close vector desnormalization

def desnorm(df_norm, df):
    df_desnorm = []
    for i in range(len(df_norm)):
        vect_desnorm = df
        df_desnorm.append(df_norm * (vect_desnorm.max()-vect_desnorm.min()) + vect_desnorm.min())
    return df_desnorm

## Vector Desnormalization

def vect_desnorm(y, y_norm):
    y_desnorm = y_norm * (y.max() - y.min()) + y.min()
    return y_desnorm

## Data frame Desnormalization

def df_desnorm(X, df):
    x_train = X
    x_desnorm_train = []
    for i in range(len(x_train)):
        x_desnorm_train.append(desnorm(x_train[i,:,0,0], df['Close']))
    return np.array(x_desnorm_train)

## Close value converter

def close_converter(close_real: object, close_ch: object) -> object:
    close_pred = []
    for i in range(len(close_ch)):
#         print(i)
#         print(close_ch[i])
        if i == 0:
            close_pred.append(close_real[-1]*(1+close_ch[i]))
        else:
            close_pred.append(close_pred[i-1]*(1+close_ch[i]))
    return np.array(close_pred)

## Predictions plotter

def predictor_plotter(X, y_real, y_pred, start, end, step, date):
    # dates = date.index
    for i in range(start, end, step):
        day = date[i]
        x_close = X[i]
        y_real_close_ch = y_real[i]
        y_pred_close_ch = y_pred[i]
        close_real = close_converter(x_close, y_real_close_ch)
        close_pred = close_converter(x_close, y_pred_close_ch)
        multi_step_plot(x_close, close_real, close_pred, day)
    return

## 5 days trend acc_dist_index

def trend_acc(X, y_real, y_pred, sign=0):
    close_real = []
    close_pred = []
    eq_trend = 0
    eq_trend_sign = 0
    hist_trend = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        x_close = X
        y_real_close_ch = y_real[i]
        y_pred_close_ch = y_pred[i]
        close_real = close_converter(x_close, y_real_close_ch)
        close_pred = close_converter(x_close, y_pred_close_ch)
#         print(close_pred)
        trend_real = close_real[-1]-close_real[0]
        trend_pred = close_pred[-1]-close_pred[0]
#         multi_step_plot(x_close, close_real, close_pred)
        if np.sign(trend_real) == np.sign(trend_pred):
            eq_trend = eq_trend +1
            hist_trend[i] = 1
        if np.sign(trend_pred) == sign:
            eq_trend_sign = eq_trend_sign +1

    return eq_trend/y_pred.shape[0], eq_trend_sign/y_pred.shape[0], hist_trend

## history calc

def history_calc(X, start, end, step):
    close_hist = []
    for i in range(start, end, step):
        x_close = X[i][0]
        close_hist.append(x_close)
    return np.array(close_hist)

## Future index value calculator

def future_calc(X, y_real,start, end, step):
    real_future = []
    for i in range(start, end, step):
        x_close = X[i][0]
        y_real_close_ch = y_real[i]
        close_real = close_converter(x_close, y_real_close_ch)
        real_future.append(close_real)
    return np.array(real_future)

## future converter trend

def futurator(df):
    first_value = []
    for i in range(len(df)):
        first_value.append(df[i][0])
    return np.array(first_value)

def predictor_plotter_future(X, y_real, y_pred, start, end, step):
    dates = X.index

    for i in range(start, end, step):
        # date_predict = len(X[0][0]) + i
        # day = dates[date_predict]
        x_close = X[i]
        y_real_close_ch = y_real[i]
        y_pred_close_ch = y_pred[i]
        close_real = close_converter(x_close, y_real_close_ch)
        close_pred = close_converter(x_close, y_pred_close_ch)
        multi_step_plot(x_close, close_real, close_pred, day)
    return

def dirmaker(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")