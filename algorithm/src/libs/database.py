from abc import abstractmethod
from ccxt import Exchange as ccxtExchange
import time
from .data_service_api import *
from bson.objectid import ObjectId

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

    
    @abstractmethod
    def write_single_account_balance(self, balance):
        pass


    @abstractmethod
    def write_multiple_account_balances(self, balances):
        pass

    @abstractmethod
    def write_single_bot_balance(self, balance):
        pass


    @abstractmethod
    def write_multiple_bots_balances(self, balances):
        pass


    @abstractmethod
    def write_operation(self, operation_type, account_id, bot_id, amount, pair):
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


    def write_accounts_balances(self, data):
        data_len = len(data)
        if data_len > 1:
            self.write_single_account_balance(data)
        elif data_len == 1:
            self.write_multiple_account_balances(data[0])
    

    def write_bots_balances(self, data):
        data_len = len(data)
        if data_len > 1:
            self.write_single_bot_balance(data)
        elif data_len == 1:
            self.write_multiple_bots_balances(data[0])


    def get_last_timestamp(self, exchange, pair, period):
        result = 0
        tohlcv = self.get_last_ohlcv(exchange, pair, period)
        if tohlcv:
            if tohlcv['timestamp']:
                result = tohlcv['timestamp']
        return result


    def preprocess_account_balance(self, balance):
        not_currency_fields = ['account_id', 'timestamp']
        res = {
            symbol: value for symbol, value in balance.items()\
                if not(symbol in not_currency_fields)
            }
        res.update({
            "timestamp": balance['timestamp'],
            "account_id": ObjectId(balance['account_id'])
            })
        return res

    
    def preprocess_bot_balance(self, balance):
        not_currency_fields = ['bot_id', 'timestamp']
        res = {symbol: value for symbol, value in balance.items()\
            if not(symbol in not_currency_fields)}
        res.update({
            "timestamp": balance['timestamp'],
            "bot_id": ObjectId(balance['bot_id'])
            })
        return res


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

