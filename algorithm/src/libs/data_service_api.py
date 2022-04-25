import requests
from .logger import *

class DataServiceAPI():
    
    def __init__(self, base_url, user_name, password, logger=None):
        self.base_url = base_url
        self.auth = (user_name, password)
        self.logger = logger


    def get_request(self, type, exchange, pair, period, from_timestamp):
        result = []
        response = requests.get(
            f"{self.base_url}/{type}/{exchange}/{pair}/{period}/{from_timestamp}",
            auth=self.auth)
        if response.status_code == 200:
            result = response.json()
        return result


    def get_ohlcv(self, exchange, pair, period, from_timestamp):
        return self.get_request('ohlcv', exchange, pair, period, from_timestamp)


    def get_close(self, exchange, pair, period, from_timestamp):
        return self.get_request('close', exchange, pair, period, from_timestamp)

    