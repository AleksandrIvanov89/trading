from .exchanges_db import *
import pymongo
import pandas as pd
from .logger import *
class MongoDB(Database):

    def __init__(
        self,
        exchange_name,
        symbol,
        username,
        password,
        socket_path="mongodb:27017",
        data_service_api=None,
        logger=None
        ):
        super().__init__(exchange_name, symbol, data_service_api, logger)
        self.client = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{socket_path}")
        self.db = self.client["trading"]
        self.db_periods = {
            period: self.db[self.exchange_name][self.symbol][period]["ohlcv"] for period in self.periods.keys()
        }


    def get_last_ohlcv(self, period):
        result = 0
        try:
            result = self.db_periods[period].find_one(
                sort=[("timestamp", pymongo.DESCENDING)]
                )
        except Exception as e:
            self._error(e)
        return result

    
    def get_ohlcv_from_db(self, period, from_timestamp=None):
        try:
            res = self.db_periods[period].find().sort([("timestamp", pymongo.ASCENDING)])\
                if from_timestamp is None else\
                    self.db_periods[period].find({'timestamp': {'$gte': from_timestamp}}).sort([("timestamp", pymongo.ASCENDING)])
            return pd.DataFrame(list(res))
        except Exception as e:
            self._error(e)
            return pd.DataFrame([])
        

    def write_single_ohlcv(self, tohlcv, period):
        try:
            if tohlcv:
                thohlcv_db = self.preprocess_ohlcv(tohlcv)
                self.db_periods[period].insert_one(thohlcv_db)
        except Exception as e:
            self._error(e)


    def write_multiple_ohlcv(self, tohlcv_list, period):
        try:
            # prepare ohlcvs for db
            tohlcv_db_list = self.preprocess_ohlcv_list(tohlcv_list)
            # write ohlcvs to db
            if len(tohlcv_db_list) > 1:
                self.db_periods[period].insert_many(tohlcv_db_list)
        except Exception as e:
            self._error(e)