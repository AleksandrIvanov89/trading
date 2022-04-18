from abc import abstractmethod
from ccxt import Exchange as ccxtExchange
import time
from .data_service_api import *


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


    @abstractmethod
    def get_ohlcv_from_db(self, period, from_timestamp=None):
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




