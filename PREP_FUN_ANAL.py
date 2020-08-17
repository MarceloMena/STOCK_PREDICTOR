import pandas as pd
import datetime as dt
import yfinance as yf

# Define today date

today = dt.date.today()
today_date = today.strftime("%Y-%m-%d")

# URL Info

gdp_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans' \
          '&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=' \
          '1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDPC1&scale=left&cosd=1' \
          '947-01-01&coed=2020-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&' \
          'lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2020-01-01&line_index=1&tran' \
          'sformation=lin&vintage_date=' + today_date + '&revision_date=' + today_date + '&nd=1947-01-01'
cp_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&' \
         'graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=11' \
         '68&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CP&scale=left&cosd=1947-01' \
         '-01&coed=2020-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&os' \
         't=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2020-01-01&line_index=1&transformati' \
         'on=lin&vintage_date=' + today_date + '&revision_date=' + today_date + '&nd=1947-01-01'
umcsent_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20' \
              'sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=1' \
              '2&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UMCSENT&sca' \
              'le=left&cosd=1952-11-01&coed=2020-05-01&line_color=%234572a7&link_values=false&line_style=solid&ma' \
              'rk_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-' \
              '01&line_index=1&transformation=lin&vintage_date=' + today_date + '&revision_date=' + today_date +\
              '&nd=1952-11-01'
wrmfsl_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20s' \
             'ans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&' \
             'width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=WRMFSL&scale=l' \
             'eft&cosd=1980-02-04&coed=2020-06-22&line_color=%234572a7&link_values=false&line_style=solid&mark_ty' \
             'pe=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Weekly%2C%20Ending%20Monday&fam=avg&fgst=lin&' \
             'fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=' + today_date +\
             '&revision_date=' + today_date + '&nd=1980-02-04'
euandh_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20s' \
             'ans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&' \
             'width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=EUANDH&scale=l' \
             'eft&cosd=1967-03-01&coed=2020-05-01&line_color=%234572a7&link_values=false&line_style=solid&mark_ty' \
             'pe=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&lin' \
             'e_index=1&transformation=lin&vintage_date='\
             + today_date + '&revision_date=' + today_date + '&nd=1967-03-01'
pandi_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sa' \
            'ns&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&wi' \
            'dth=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=PANDI&scale=left&' \
            'cosd=1967-03-01&coed=2020-05-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=no' \
            'ne&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_inde' \
            'x=1&transformation=lin&vintage_date=' + today_date + '&revision_date=' + today_date + '&nd=1967-03-01'
anfci_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sa' \
            'ns&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&wi' \
            'dth=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=ANFCI&scale=left&' \
            'cosd=1971-01-08&coed=2020-06-26&line_color=%234572a7&link_values=false&line_style=solid&mark_type=no' \
            'ne&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Weekly%2C%20Ending%20Friday&fam=avg&fgst=lin&fgsnd=' \
            '2020-02-01&line_index=1&transformation=lin&vintage_date=' + today_date + \
            '&revision_date=' + today_date + '&nd=1971-01-08'
indpro_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20' \
             'sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=1' \
             '2&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=INDPRO&scal' \
             'e=left&cosd=1919-01-01&coed=2020-05-01&line_color=%234572a7&link_values=false&line_style=solid&mar' \
             'k_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2009-06-0' \
             '1&line_index=1&transformation=lin&vintage_date=' + today_date + \
             '&revision_date=' + today_date + '&nd=1919-01-01'
dcoilwtico_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=ope' \
                 'n%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=' \
                 '12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id' \
                 '=DCOILWTICO&scale=left&cosd=2015-06-29&coed=2020-06-29&line_color=%234572a7&link_values=false&' \
                 'line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fg' \
                 'st=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=' + today_date + \
                 '&revision_date=' + today_date + '&nd=1986-01-02'
AAII_url = 'https://www.quandl.com/api/v3/datasets/AAII/AAII_SENTIMENT.csv?api_key=scz2vL5_ZqWqdkJ8UWrL'

# GDP

US_GDP = pd.read_csv(gdp_url)
US_GDP['DATE'] = pd.to_datetime(US_GDP['DATE'])

# CP

CORP_PROFIT = pd.read_csv(cp_url)
CORP_PROFIT['DATE'] = pd.to_datetime(CORP_PROFIT['DATE'])
SP500_FUN_AN = pd.merge(US_GDP, CORP_PROFIT, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Consumer sentiment

CONS_SENT = pd.read_csv(umcsent_url)
CONS_SENT['DATE'] = pd.to_datetime(CONS_SENT['DATE'])
CONS_SENT['UMCSENT'] = pd.to_numeric(CONS_SENT['UMCSENT'], errors='coerce')
SP500_FUN_AN = pd.merge(SP500_FUN_AN, CONS_SENT, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Retail money funds

RETAIL_FUNDS = pd.read_csv(wrmfsl_url)
RETAIL_FUNDS['DATE'] = pd.to_datetime(RETAIL_FUNDS['DATE'])
SP500_FUN_AN = pd.merge(SP500_FUN_AN, RETAIL_FUNDS, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Unemployment and work hours

UN_EMPLOY_HOU = pd.read_csv(euandh_url)
UN_EMPLOY_HOU['DATE'] = pd.to_datetime(UN_EMPLOY_HOU['DATE'])
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')
SP500_FUN_AN = pd.merge(SP500_FUN_AN, UN_EMPLOY_HOU, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Production Income

PROD_INCOME = pd.read_csv(pandi_url)
PROD_INCOME['DATE'] = pd.to_datetime(PROD_INCOME['DATE'])
SP500_FUN_AN = pd.merge(SP500_FUN_AN, PROD_INCOME, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Financial conditions

FIN_COND = pd.read_csv(anfci_url)
FIN_COND['DATE'] = pd.to_datetime(FIN_COND['DATE'])
SP500_FUN_AN = pd.merge(SP500_FUN_AN, FIN_COND, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Production index

IND_PROD = pd.read_csv(indpro_url)
IND_PROD['DATE'] = pd.to_datetime(IND_PROD['DATE'])
SP500_FUN_AN = pd.merge(SP500_FUN_AN, IND_PROD, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Gold

GOLD_PRICE = yf.Ticker("GC=F")
GOLD_PRICE = GOLD_PRICE.history(period="1y")
GOLD_PRICE.reset_index(inplace=True)
GOLD_PRICE.rename(columns={'Close': 'GOLD', 'Date': 'DATE'}, inplace=True)
GOLD_PRICE = GOLD_PRICE.drop(['Open', 'High', 'Low', 'Dividends', 'Volume', 'Stock Splits'], axis=1)
SP500_FUN_AN = pd.merge(SP500_FUN_AN, GOLD_PRICE, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Oil

OIL_WTI_PRICE = pd.read_csv(dcoilwtico_url)
OIL_WTI_PRICE['DATE'] = pd.to_datetime(OIL_WTI_PRICE['DATE'])
OIL_WTI_PRICE['DCOILWTICO'] = pd.to_numeric(OIL_WTI_PRICE['DCOILWTICO'], errors='coerce')
SP500_FUN_AN = pd.merge(SP500_FUN_AN, OIL_WTI_PRICE, on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')

# Investor Sentiment

INVES_MARKET = pd.read_csv(AAII_url)
INVES_MARKET['Date'] = pd.to_datetime(INVES_MARKET['Date'])
INVES_MARKET = INVES_MARKET.rename(columns={"Date": "DATE"})
SP500_FUN_AN = pd.merge(SP500_FUN_AN, INVES_MARKET[['DATE', 'Bullish', 'Neutral', 'Bearish', 'S&P 500 Weekly Close']],
                        on='DATE', how='outer')
SP500_FUN_AN = SP500_FUN_AN.sort_values('DATE')
SP500_FUN_AN = SP500_FUN_AN.set_index('DATE')

# NaN Values

SP500_FUN_AN = SP500_FUN_AN.fillna(method='ffill')

start = dt.datetime(2019, 6, 1)
end = dt.datetime.now()
SP500_plot = SP500_FUN_AN.loc[start:end]

SP500_plot.to_csv('Future/STOCK_FUN_AN')

# SP500_plot.to_csv('/home/marcelo/Downloads/'+stock_name+'.csv')

print(SP500_plot.tail())
