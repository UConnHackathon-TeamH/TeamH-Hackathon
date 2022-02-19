import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DNN_Model import *

# Gather data
header_list = ["pair", "price", "volume", "timestamp"]
data_2 = pd.read_csv("/home/ubuntu/tb-hackathon/xbt.usd.2018.csv", names = header_list)
data_2 = data_2.truncate(after = 10000000)

symbol = data_2.columns[1]
data_2['returns'] = np.log(data_2[symbol] / data_2[symbol].shift())
window = 50

# Add features
# Adding Label/Features
df = data_2.copy()
df['dir'] = np.where(df['returns'] > 0,1,0)
df["sma"] = df[symbol].rolling(window).mean() - df[symbol].rolling(150).mean()
df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window) .std()
df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1
df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1
df["mom"] = df["returns"].rolling(3).mean()
df["vol"] = df["returns"].rolling(window).std()
df.dropna(inplace = True)

lags = 5
cols = []
features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

for f in features:
    for lag in range(1, lags + 1):
        col = "{}_lag{}".format(f, lag)
        df[col] = df[f].shift(lag)
        cols.append(col)
df.dropna(inplace = True)

# Split data
split = int(len(df)*0.66)

# Train data
train = df.iloc[:split].copy()
mu, std = train.mean(), train.std()
train_s = (train - mu) / std

set_seeds(100)
model = create_model(hl=3, hu=50, dropout = True, input_dim=len(cols))
model.fit(x=train_s[cols], y = train["dir"], epochs=50, verbose=False,
        validation_split=0.2, shuffle = False, class_weight=cw(train))

model.save("DNN_Model_BTC")