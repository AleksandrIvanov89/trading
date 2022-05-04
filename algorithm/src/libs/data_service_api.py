import requests
import json
from .logger import *

class DataServiceAPI():
    
    def __init__(self, base_url, user_name, password, logger=None):
        self.base_url = base_url
        self.auth = (user_name, password)
        self.logger = logger


    def get_request(self, url):
        result = []
        response = requests.get(
            f"{self.base_url}/{url}",
            auth=self.auth
            )
        if response.status_code == 200:
            result = response.json()
        return result

    
    def post_request(self, url, data={}):
        result = []
        response = requests.post(
            f"{self.base_url}/{url}",
            data=json.dumps(data),
            auth=self.auth
            )
        if response.status_code == 200:
            result = response.json()
        return result


    def get_ohlcv_request(self, type, exchange, pair, period, from_timestamp):
        return self.get_request(
            f"{type}/{exchange}/{pair}/{period}/{from_timestamp}"
            )


    def get_ohlcv(self, exchange, pair, period, from_timestamp):
        return self.get_ohlcv_request(
            'ohlcv',
            exchange,
            pair,
            period,
            from_timestamp
            )


    def get_close(self, exchange, pair, period, from_timestamp):
        return self.get_ohlcv_request(
            'close',
            exchange,
            pair,
            period,
            from_timestamp
            )

    
    def get_account_balances(self, account_id):
        return self.get_request(f"account_balances/{account_id}")

    
    def get_all_accounts_balances(self):
        return self.get_request(f"all_accounts_balances")

    
    def get_all_bots_balances(self):
        return self.get_request(f"all_bots_balances")


    def post_operation(self, operation_type, bot_id, amount):
        return self.post_request(
            f"make_operation/{operation_type}/{bot_id}/{amount}",
            {
                'operation_type': operation_type,
                'bot_id': bot_id,
                'amount': amount
                })
