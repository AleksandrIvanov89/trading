from .database import *
import pymongo
from bson.objectid import ObjectId
import pandas as pd

class MongoDB(Database):

    def __init__(
        self,
        username,
        password,
        socket_path="mongodb:27017"
    ):
        super().__init__()
        self.client = pymongo.MongoClient(f"mongodb://{username}:{password}@{socket_path}")
        self.db = self.client["trading"]


    def get_account(self, account_id):
        res = dict(
            self.db['accounts'].find_one(
                {"_id": ObjectId(account_id)}))
        return res['exchange'], res['bots']

    
    def get_bot(self, bot_id):
        res = dict(
            self.db['bots'].find_one(
                {"_id": ObjectId(bot_id)}))
        return res['exchange_id'], res['symbol'], res['algorithm_id'], res['type'], res['state']

    
    def get_exchange(self, exchange_id):
        res = dict(
            self.db['exchanges'].find_one(
                {"_id": ObjectId(exchange_id)}))
        symbols = res['active_symbols'] if 'active_symbols' in res.keys() else []
        return res['name'], res['ccxt_id'], symbols


    def get_active_exchanges(self):
        resp = list(
            self.db["exchanges"].find(
                {'active_symbols': {"$exists": True}}))
        res = []
        for exchange_i in resp:
            for symbol in exchange_i['active_symbols']:
                res += [{'exchange': exchange_i['ccxt_id'], 'symbol': symbol}]
        return res

    def get_account_balances_from_db(self, account_id, from_timestamp=None):
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
            print(f"Error:\n{e}")
        return pd.DataFrame(res)

    
    def get_last_balance_from_db(self, account_id):
        res = {}
        try:
            temp = self.db["account_balances"].find_one(
                    {'account_id': ObjectId(account_id)},
                    sort=[("timestamp", pymongo.DESCENDING)])
            res = {key: temp[key] for key in temp.keys() if not(key in ['_id', 'account_id', 'timestamp'])}
        except Exception as e:
            print(f"Error:\n{e}")
        return res