from .exchanges_db import *
import pymongo
import pandas as pd
class MongoDB(Database):

    def __init__(
        self,
        exchange_name,
        symbol,
        username,
        password,
        socket_path="mongodb:27017",
        data_service_api=None
    ):
        super().__init__(exchange_name, symbol, data_service_api)
        self.client = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{socket_path}")
        self.db = self.client["trading"]
        self.db_periods = {
            period: self.db[self.exchange_name][self.symbol][period]["ohlcv"] for period in self.periods.keys()
        }


    def get_active_exchanges(self):
        resp = list(self.db["exchanges"].find({'active_symbols': {"$exists": True}}))
        res = []
        for exchange_i in resp:
            for symbol in exchange_i['active_symbols']:
                res += [{'exchange': exchange_i['ccxt_id'], 'symbol': symbol}]
        return res



    def get_last_ohlcv(self, period):
        try:
            result = self.db_periods[period].find_one(
                sort=[("timestamp", pymongo.DESCENDING)])
            return result
        except Exception as e:
            print(f"Error:\n{e}")
            return 0

    
    def get_ohlcv_from_db(self, period, from_timestamp=None):
        try:
            res = self.db_periods[period].find().sort([("timestamp", pymongo.ASCENDING)])\
                if from_timestamp is None else\
                    self.db_periods[period].find({'timestamp': {'$gte': from_timestamp}}).sort([("timestamp", pymongo.ASCENDING)])
            return pd.DataFrame(list(res))
        except Exception as e:
            print(f"Error:\n{e}")
            return pd.DataFrame([])
        

    def write_single_ohlcv(self, tohlcv, period):
        if tohlcv:
            thohlcv_db = self.preprocess_ohlcv(tohlcv)
            self.db_periods[period].insert_one(thohlcv_db)


    def write_multiple_ohlcv(self, tohlcv_list, period):
        # prepare ohlcvs for db
        tohlcv_db_list = self.preprocess_ohlcv_list(tohlcv_list)
        # write ohlcvs to db
        if len(tohlcv_db_list) > 1:
            self.db_periods[period].insert_many(tohlcv_db_list)