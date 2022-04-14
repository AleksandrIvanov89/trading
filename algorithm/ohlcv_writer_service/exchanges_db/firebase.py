from .exchanges_db import *
import firebase_admin
from firebase_admin import credentials, firestore

class Firebase(Database):
    
    def __init__(self, credentials_path):
        creds = credentials.Certificate(credentials_path)
        self.client = firebase_admin.initialize_app(creds)
        self.db = firestore.client()  # connect to Firestore database


    def get_last_ohlcv(self, period):
        try:
            #Firebase request#result = self.db[self.exchange_name][self.symbol][period]["ohlcv"].find_one(sort=[("timestamp", pymongo.DESCENDING)])
            result = 0
            return result
        except Exception as e:
            print(f"Error:\n{e}")
            return 0


    def write_single_ohlcv(self, tohlcv, period):
        if tohlcv:
            thohlcv_db = self.preprocess_ohlcv(tohlcv)
            #Firebase request#self.db[self.exchange_name][self.symbol][period]["ohlcv"].insert_one(tohlcv)
            pass


    def write_multiple_ohlcv(self, tohlcv_list, period):
        # prepare ohlcvs for db
        tohlcv_db_list = self.preprocess_ohlcv_list(tohlcv_list)
        # write ohlcvs to db
        if len(tohlcv_db_list) > 1:
            #Firebase request#self.db[self.exchange_name][self.symbol][period]["ohlcv"].insert_many(tohlcv_db_list)
            pass
