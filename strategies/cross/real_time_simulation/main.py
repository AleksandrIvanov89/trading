import os
import json
import ccxt
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

class Database():

    def __init__(self, credentials_path, bot_id):
        self.cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(self.cred)
        self.bot_id = bot_id
        self.db = firestore.client()  # connect to Firestore database
        self.document = self.db.collection(u'bots').document(bot_id)


    def load_params(self):
        self.params = self.document.get().to_dict()


    def is_first_launch(self):
        return len(self.document.collection(u'states').limit(1).get()) == 0

class Exchange():
    sec = 1000
    min = 60000
    buffer = 10
    tohlcv_columns = ["timestamp", "open", "high", "low", "close", "volume", "period"]

    def __init__(self, exchange_id, symbol, period='1m'):
        self.id = exchange_id
        self.symbol = symbol
        self.period = period


    def connect(self):
        self.exchange = getattr(ccxt, self.id)({'enableRateLimit': True, })
        self.markets = self.exchange.load_markets()


    def get_time(self):
        return self.exchange.milliseconds()


    def load_ohlcv(self, from_timestamp=0):
        if from_timestamp == 0:
            from_timestamp = self.last_read_timestamp

        prev_from_timestamp = 0
        tholcv_list = []
        while prev_from_timestamp != from_timestamp:
            try:
                tohlcv_list_temp = self.exchange.fetch_ohlcv(self.symbol, self.period, from_timestamp)
                # loop variables
                prev_from_timestamp = from_timestamp
                if len(tohlcv_list_temp) > 0:
                    if len(tholcv_list) == 0:
                        tholcv_list = tohlcv_list_temp
                    else:
                        if tholcv_list[-1][0] != tohlcv_list_temp[0][0]:
                            tholcv_list += tohlcv_list_temp    
                    from_timestamp = tohlcv_list_temp[-1][0]

            except Exception as e:
                print("Error: ", e)
        self.last_read_timestamp = from_timestamp
        return tholcv_list
    
    def first_ohlcv_load(self, period):
        from_timestamp = self.get_time() - self.min * (period + self.buffer)
        self.load_ohlcv(from_timestamp)


class Exchange_Simulator(Exchange):
    pass


class Algorithm():

    def __init__(self, database, exchange):
        self.database = database
        self.exchange = exchange
        self.exchange.connect()

    def initialize():
        pass

    def process(self):
        pass


class Cross_MA_Algorithm(Algorithm):
    def __init__(self, database, exchange):
        super().__init__(database, exchange)

        self.window_short = self.database.params['window_short']
        self.window_long = self.database.params['window_long']

    def initialize(self):
        self.ma = pd.DataFrame(
            self.exchange.first_ohlcv_load(self.window_long),
            columns=self.exchange.tohlcv_columns)
        print(self.ma.head(5))
        print(self.ma.tail(5))

        self.ma['window_short'] = self.ma['close'].rolling(window=self.window_short).mean()
        self.ma['window_long'] = self.ma['close'].rolling(window=self.window_long).mean()
        print(self.ma.head(5))
        print(self.ma.tail(5))
        

    def process(self):
        res = self.exchange.load_ohlcv()
        if (res != []):
            df_len = self.ma.shape[0]
            new_df = pd.DataFrame(res, columns=self.exchange.tohlcv_columns)
            print(new_df)
            self.ma = pd.concat([self.ma, new_df], ignore_index=True)
            self.ma.reset_index(inplace=True)
            self.ma.drop(range(self.ma.shape[0] - df_len))
            print(self.ma.head(5))
            print(self.ma.tail(5))

            


def init_algo(database):
    #if database.params['type'] == 'simulation':
    exchange = Exchange_Simulator(database.params['exchange'], database.params['symbol'], database.params['period'])
    #if database.params['algorithm'] == 'cross_ma':
    return Cross_MA_Algorithm(database, exchange)



def main():
    database = Database(os.environ.get("FIREBASE_CREDENTIALS_PATH"), os.environ.get("FIREBASE_BOT_ID"))
    database.load_params()
    
    algo = init_algo(database)
    algo.initialize()
    while True:
        algo.process()
    

if __name__ == '__main__':
    main()