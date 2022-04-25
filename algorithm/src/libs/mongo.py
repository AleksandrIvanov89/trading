from .database import *
from .logger import *
import pymongo
from bson.objectid import ObjectId
import pandas as pd
import inspect

class MongoDB(Database):

    def __init__(
        self,
        username,
        password,
        socket_path="mongodb:27017",
        logger=None
    ):
        super().__init__(logger)
        self.client = pymongo.MongoClient(
            f"mongodb://{username}:{password}@{socket_path}"
            )
        self.db = self.client["trading"]


    def get_account(self, account_id):
        res = dict(
            self.db['accounts'].find_one(
                {"_id": ObjectId(account_id)}
                ))
        return res['exchange'], res['bots'], res['type']

    
    def get_all_accounts(self):
        return list(self.db['accounts'].find())

    
    def get_bot(self, bot_id):
        res = dict(
            self.db['bots'].find_one(
                {"_id": ObjectId(bot_id)}
                ))
        return res['exchange_id'], res['symbol'], res['algorithm_id'], res['type'], res['state']

    
    def get_exchange(self, exchange_id):
        res = dict(
            self.db['exchanges'].find_one(
                {"_id": ObjectId(exchange_id)}
                ))
        symbols = res['active_symbols'] if 'active_symbols' in res.keys() else []
        return res['name'], res['ccxt_id'][-1], symbols


    def get_active_exchanges(self):
        resp = list(
            self.db["exchanges"].find(
                {'active_symbols': {"$exists": True}}
                ))
        return [
            {
                'name': exchange_i['name'],
                'id': exchange_i['_id'],
                'symbol': exchange_i['active_symbols']
                } for exchange_i in resp]


    def get_last_ohlcv(self, exchange, pair, period):
        result = 0
        try:
            result = self.db[exchange][pair][period].find_one(
                sort=[("timestamp", pymongo.DESCENDING)]
                )
        except Exception as e:
            log(
                f"Exception in MongoDB:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )
        return result
    

    def get_ohlcv(self, exchange, pair, period, from_timestamp=None):
        try:
            res = self.db[exchange][pair][period]["ohlcv"].find().sort(
                [("timestamp", pymongo.ASCENDING)]
                ) if from_timestamp is None else\
                    self.db[exchange][pair][period]["ohlcv"].find(
                        {'timestamp': {'$gte': from_timestamp}}
                        ).sort(
                            [("timestamp", pymongo.ASCENDING)]
                            )
            return pd.DataFrame(list(res))
        except Exception as e:
            log(
                f"Exception in MongoDB:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )
            return pd.DataFrame([])
    

    def write_single_ohlcv(self, exchange, pair, period, tohlcv):
        try:
            thohlcv_db = self.preprocess_ohlcv(tohlcv)
            self.db[exchange][pair][period]["ohlcv"].insert_one(thohlcv_db)
        except Exception as e:
            log(
                f"Exception in MongoDB:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )


    def write_multiple_ohlcv(self, exchange, pair, period, tohlcv_list):
        try:
            # prepare ohlcvs for db
            tohlcv_db_list = self.preprocess_ohlcv_list(tohlcv_list)
            # write ohlcvs to db
            self.db[exchange][pair][period]["ohlcv"].insert_many(tohlcv_db_list)
        except Exception as e:
            log(
                f"Exception in MongoDB:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )


    def get_account_balances(self, account_id, from_timestamp=None):
        res = []
        try:
            if from_timestamp is None:
                res = list(
                    self.db["account_balances"].find(
                        {'account_id': ObjectId(account_id)}
                        ).sort(
                            [("timestamp", pymongo.ASCENDING)]))
            else:
                res = list(
                    self.db["account_balances"].find(
                        {"$and":[
                            {'account_id': ObjectId(account_id)},
                            {'timestamp': {'$gte': from_timestamp}}]}
                        ).sort(
                            [("timestamp", pymongo.ASCENDING)]))
        except Exception as e:
            log(
                f"Exception in MongoDB:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )
        return pd.DataFrame(res)

    
    def get_last_balance(self, account_id):
        res = {}
        not_currency_fields = ['_id', 'account_id', 'timestamp']
        try:
            temp = self.db["account_balances"].find_one(
                    {'account_id': ObjectId(account_id)},
                    sort=[("timestamp", pymongo.DESCENDING)]
                    )
            res = {
                key: temp[key] for key in temp.keys() if not(key in not_currency_fields)
                }
        except Exception as e:
            log(
                f"Exception in MongoDB:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )
        return res