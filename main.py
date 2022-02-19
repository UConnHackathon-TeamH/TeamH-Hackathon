from boto3 import resource
from dataclasses import dataclass
from decimal import Decimal
from dotenv import load_dotenv
from os import getenv
from s3_helper import CSVStream
from typing import Any
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

load_dotenv()

BUY = "buy"
SELL = "sell"

BUCKET = getenv("BUCKET_NAME")

XBT_2018_KEY = "xbt.usd.2018"
XBT_2020_KEY = "xbt.usd.2020"

ETH_2018_KEY = "eth.usd.2018"
ETH_2020_KEY = "eth.usd.2020"

S3 = resource("s3")
SELECT_ALL_QUERY = 'SELECT * FROM S3Object'

# Example s3 SELECT Query to Filter Data Stream
#
# The where clause fields refer to the timestamp column in the csv row.
# To filter the month of February, for example, (start: 1517443200, end: 1519862400) 2018
#                                               (Feb-01 00:00:00  , Mar-01 00:00:00) 2018
#
# QUERY = '''\
#     SELECT *
#     FROM S3Object s
#     WHERE CAST(s._4 AS DECIMAL) >= 1514764800
#       AND CAST(s._4 AS DECIMAL) < 1514764802
# '''

STREAM = CSVStream(
    'simple',
    S3.meta.client,
    key=XBT_2018_KEY,
    bucket=BUCKET,
    expression=SELECT_ALL_QUERY,
)
header_list = ["eth-usd", "price", "volume", "timestamp"]
data_2 = pd.read_csv("eth.usd.2018.csv", names = header_list)
data_2 = data_2.truncate(after = 100000)

import keras
model = keras.models.load_model("DNN_model")

import pickle
params = pickle.load(open("params.pkl", "rb"))
mu = params["mu"]
std = params['std']

@dataclass
class Trade:
    trade_type: str # BUY | SELL
    base: str
    volume: Decimal

class Predict:
    def __init__(self, instrument, window, lags, model, mu, std, start_time=0, user_input=None):
        self.user_input = user_input
        self.start_time = start_time
        self.instrument = instrument
        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        self.raw_data = None            # review later
        #self.tick_data = pd.Dataframe() # review later
        self.remaining_fund = 1000000
        self.eth_position = 0
        self.btc_position = 0

    def define_strategy(self):
        df = data_2.copy()
        df = df.append(self.user_input, ignore_index=True) # append the latest tick
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df['dir'] = np.where(df['returns'] > 0,1,0)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)
        

        # create logs
        self.cols = []
        features = ["dir", "sma", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags +1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace=True)

        # standardization
        self.mu = df.mean()
        self.std = df.std()
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.model.predict(df_s[self.cols])

        # Determine position
        #df = df.loc[self.start_time:].copy() # remove all historical data
        df["position"] = np.where(df.proba < 0.47, 1, np.nan)
        df["position"] = np.where(df.proba > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0)

        self.data = df.copy()

    def add_input(self, user_input):
        self.user_input = user_input

    def get_recent_df_row(self):
        self.define_strategy()
        return self.data['proba'].iloc[-1:]

predict = Predict(instrument='price', window=50, lags=5, model=model, mu=mu, std=std)

def algorithm(csv_row: str):
    """ Trading Algorithm

    Add your logic to this function. This function will simulate a streaming
    interface with exchange trade data. This function will be called for each
    data row received from the stream.

    The context object will persist between iterations of your algorithm.

    Args:
        csv_row (str): one exchange trade (format: "exchange pair", "price", "volume", "timestamp")
        context (dict[str, Any]): a context that will survive each iteration of the algorithm

    Generator:
        response (dict): "Fill"-type object with information for the current and unfilled trades
    
    Yield (None | Trade | [Trade]): a trade order/s; None indicates no trade action
    """
    lst = [csv_row.split(",")] # lst = ['eth-usd','300','1','1265667778']
    lst[0][1] = float(lst[0][1])
    lst[0][2] = float(lst[0][2])
    lst[0][3] = float(lst[0][3])
    csv_row = pd.DataFrame(lst, columns=['pair','price', 'volume','timestamp'])
    #predict.add_time(start_time)
    predict.add_input(csv_row) # list: ["eth-usd","172","1","151519237548023"]
    probability = predict.get_recent_df_row()
    ticker = lst[0][0][5:8]

    #print(probability)
    if probability > 0.53:
        if predict.remaining_fund >= lst[0][1]:
            response = yield Trade("BUY", ticker, Decimal(1))
            if ticker == 'btc':
                predict.btc_position += 1
            else:
                predict.eth_position += 1
        else:
            response = yield None
    elif probability < 0.47:
        if ticker == 'btc':
            if predict.btc_position >= 1:
                response = yield Trade("SELL", ticker, Decimal(1))
            else:
                response = yield None
        if ticker == 'eth':
            if predict.eth_position >= 1:
                response = yield Trade("SELL", ticker, Decimal(1))
            else:
                response = yield None
    else:
        response = yield None
    
    #response = yield None # example: Trade(BUY, 'xbt', Decimal(1))

    # algorithm clean-up/error handling...

if __name__ == '__main__':
    # example to stream data
    for row in STREAM.iter_records():
        algorithm(row)

# Example Interaction
#
# Given the following incoming trades, each line represents one csv row:
#   (1) okfq-xbt-usd,14682.26,2,1514765115
#   (2) okf1-xbt-usd,13793.65,2,1514765115
#   (3) stmp-xbt-usd,13789.01,0.00152381,1514765115
#
# When you receive trade 1 through to your algorithm, if you decide to make
# a BUY trade for 3 xbt, the order will start to fill in the following steps
#   [1] 1 unit xbt from trade 1 (%50 available volume from the trade data)
#   [2] 1 unit xbt from trade 2
#   [3] receiving trade 3, you decide to put in another BUY trade:
#       i. Trade will be rejected, because we have not finished filling your 
#          previous trade
#       ii. The fill object will contain additional fields with error data
#           a. "error_code", which will be "rejected"; and
#           b. "error_msg", description why the trade was rejected.
#
# Responses during these iterations:
#   [1] success resulting in:
#       {
#           "price": 14682.26,
#           "volume": 1,
#           "unfilled": {"xbt": 2, "eth": 0 }
#       }
#   [2]
#       {
#           "price": 13793.65,
#           "volume": 1,
#           "unfilled": {"xbt": 1, "eth": 0 }
#       }
#   [3]
#       {
#           "price": 13789.01,
#           "volume": 0.000761905,
#           "error_code": "rejected",
#           "error_msg": "filling trade in progress",
#           "unfilled": {"xbt": 0.999238095, "eth": 0 }
#       }
#
# In step 3, the new trade order that you submitted is rejected; however,
# we will continue to fill that order that was already in progress, so
# the price and volume are CONFIRMED in that payload.
