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

    def define_strategy(self):
        df = data_2.copy()
        df = df.append(self.user_input) # append the latest tick
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

    def add_time(self, time):
        self.start_time = time

    def add_input(self, user_input):
        self.user_input = user_input

    def get_recent_df_row(self):
        self.define_strategy()
        return self.data['proba']

    def update_up(self, up1, up2, up3, up4):
        self.up_1 = up1
        self.up_2 = up2
        self.up_3 = up3
        self.up_4 = up4