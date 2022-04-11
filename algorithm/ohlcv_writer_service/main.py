import os
import json
import time
import requests
from abc import abstractmethod
from ccxt import Exchange as ccxtExchange
import pymongo
import firebase_admin
from firebase_admin import credentials, firestore


class DataServiceAPI():

    def __init__(self, base_url, user_name, password):
        self.base_url = base_url
        self.auth = (user_name, password)


    def get_request(self, type, period, from_timestamp):
        return requests.get(
            f"{self.base_url}/{type}/{period}/{from_timestamp}",
            auth=self.auth)


    def get_ohlcv(self, period, from_timestamp):
        return self.get_request('ohlcv', period, from_timestamp)


    def get_close(self, period, from_timestamp):
        return self.get_request('close', period, from_timestamp)


class Database():

    periods = {
        '1m': 60000,
        '1h': 3600000,
        '1d': 86400000
    }

    tohlcv_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    last_timestamp = {
        '1m': 0,
        '1h': 0,
        '1d': 0
    }

    last_read = {
        '1m': 0,
        '1h': 0,
        '1d': 0
    }


    def __init__(self, exchange_name, symbol, data_service_api=None):
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.data_service_api = data_service_api


    def get_ohlcv_from_data_service(self, period, from_timestamp):
        result = []
        if self.data_service_api:
            response = self.data_service_api.get_ohlcv(period, from_timestamp)
            if response.status_code == 200:
                result = response.json()
        return result


    @abstractmethod
    def get_last_ohlcv(self, period):
        return None


    def get_last_timestamp(self, period):
        result = 0
        tohlcv = self.get_last_ohlcv(period)
        if tohlcv:
            if tohlcv['timestamp']:
                result = tohlcv['timestamp']
        return result


    def write_ohlcv(self, tohlcv_list, period):
        tohlcv_len = len(tohlcv_list)
        if tohlcv_len > 0:
            try:
                if tohlcv_len == 1:
                    self.write_single_ohlcv(tohlcv_list[0], period)
                else:
                    self.write_multiple_ohlcv(tohlcv_list, period)
                self.last_timestamp[period] = tohlcv_list[-1]['timestamp']
            except Exception as e:
                print(f"Error:\n{e}")


    @abstractmethod
    def write_single_ohlcv(self, tohlcv, period):
        pass


    @abstractmethod
    def write_multiple_ohlcv(self, tohlcv_list, period):
        pass


    def update_ohlcv(self, period):
        try:
            if self.last_timestamp[period] == 0:
                self.last_timestamp[period] = self.get_last_timestamp(period)

            ohlcv_list = self.get_ohlcv_from_data_service(
                period,
                self.last_timestamp[period])

            self.write_ohlcv(ohlcv_list, period)
        except Exception as e:
            print(f"Error:\n{e}")
            return 0


    def update_ohlcv_loop(self):
        while True:
            for period, step in self.periods.items():
                cur_time = int(time.time() * 1000)
                if self.last_read[period] + step < cur_time:
                    self.update_ohlcv(period)
                    self.last_read[period] = cur_time


    def update_ohlcvs_all_periods(self):
        for period in self.periods.keys():
            self.update_ohlcv(period)


    def preprocess_ohlcv(self, tohlcv):
        return {
            "timestamp": tohlcv['timestamp'],
            "datetime": ccxtExchange.iso8601(tohlcv['timestamp']),
            "open": tohlcv['open'],
            "high": tohlcv['high'],
            "low": tohlcv['low'],
            "close": tohlcv['close'],
            "volume": tohlcv['volume']
        }


    def preprocess_ohlcv_list(self, tohlcv_list):
        return [self.preprocess_ohlcv(tohlcv) for tohlcv in tohlcv_list]


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


    def get_last_ohlcv(self, period):
        try:
            result = self.db_periods[period].find_one(
                sort=[("timestamp", pymongo.DESCENDING)])
            return result
        except Exception as e:
            print(f"Error:\n{e}")
            return 0


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


def load_config():
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange")  # name from ccxt library
        symbol = json_data.get("symbol")  # format - BTC/USDT
        return exchange_name, symbol


def main():
    exchange_name, symbol = load_config()
    #os.environ.get("FIREBASE_CREDENTIALS_PATH"), os.environ.get("FIREBASE_BOT_ID")
    mongo_username = os.environ.get("MONGO_USERNAME")
    mongo_password = os.environ.get("MONGO_PASSWORD")
    rest_api_user = os.environ.get("REST_API_USER")
    rest_api_password = os.environ.get("REST_API_PASSWORD")
    rest_api_base_url = os.environ.get("REST_API_BASE_URL")
    data_service_api = DataServiceAPI(
        rest_api_base_url, rest_api_user, rest_api_password)
    mongodb = MongoDB(exchange_name, symbol, mongo_username,
                      mongo_password, "mongodb:27017", data_service_api)

    mongodb.update_ohlcv_loop()


if __name__ == '__main__':
    main()
