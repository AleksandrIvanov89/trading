from abc import abstractmethod
from ccxt import Exchange as ccxtExchange
import time
from .data_service_api import *


class Database():

    def __init__(self, logger=None):
        self.logger = logger


    @abstractmethod
    def get_last_ohlcv(self, exchange, pair, period):
        return None


    @abstractmethod
    def get_ohlcv(self, exchange, pair, period, from_timestamp=None):
        return None

    
    @abstractmethod
    def write_single_ohlcv(self, exchange, pair, period, tohlcv):
        pass


    @abstractmethod
    def write_multiple_ohlcv(self, exchange, pair, period, tohlcv_list):
        pass

    
    def write_ohlcv(self, exchange, pair, period, tohlcv_list):
        list_len = len(tohlcv_list)
        if list_len > 1:
            self.write_multiple_ohlcv(
                exchange,
                pair,
                period,
                tohlcv_list
                )
        elif list_len == 1:
            self.write_single_ohlcv(
                exchange,
                pair,
                period,
                tohlcv_list[0]
                )


    def get_last_timestamp(self, exchange, pair, period):
        result = 0
        tohlcv = self.get_last_ohlcv(exchange, pair, period)
        if tohlcv:
            if tohlcv['timestamp']:
                result = tohlcv['timestamp']
        return result
    

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