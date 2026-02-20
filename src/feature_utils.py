
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['AAPL', 'NVDA', 'TSLA','AMD','META' ]
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'NASDAQCOM', 'VIXCLS']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)



    return_period = 5
    Y = stk_data['Adj Close']['NVDA'].pct_change(return_period).shift(-return_period)
    Y.name = 'NVDA_Future_Return'
    
    # Features
    X = pd.DataFrame({
        'AAPL_return':    stk_data['Adj Close']['AAPL'].pct_change(),
        'TSLA_return':    stk_data['Adj Close']['TSLA'].pct_change(),
        'AMD_return':     stk_data['Adj Close']['AMD'].pct_change(),
        'META_return':    stk_data['Adj Close']['META'].pct_change(),
        'range':          (stk_data['High']['NVDA'] - stk_data['Low']['NVDA']) / stk_data['Close']['NVDA'],
        'gap':            (stk_data['Open']['NVDA'] - stk_data['Close']['NVDA'].shift(1)) / stk_data['Close']['NVDA'].shift(1),
        'momentum_10':    stk_data['Adj Close']['NVDA'] / stk_data['Adj Close']['NVDA'].shift(10) - 1,
        'volatility_14':  stk_data['Adj Close']['NVDA'].pct_change().rolling(14).std(),
    })
    
    # Combine and drop missing values
    dataset = pd.concat([Y, X], axis=1).dropna()
    

    #Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
    #Y.name = Y.name[-1]+'_Future'
    
    #X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
    #X1.columns = X1.columns.droplevel()
    #X2 = np.log(ccy_data).diff(return_period)
    #X3 = np.log(idx_data).diff(return_period)

    #X = pd.concat([X1, X2, X3], axis=1)
    
    #dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]

    Y = dataset['NVDA_Future_Return']
    X = dataset.drop(columns=['NVDA_Future_Return'])

    #Y = dataset.loc[:, Y.name]
    #X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'
    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df


