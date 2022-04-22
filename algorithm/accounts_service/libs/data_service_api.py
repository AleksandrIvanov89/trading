import requests
from .logger import *

class DataServiceAPI():
    
    def __init__(self, base_url, user_name, password, logger=None):
        self.base_url = base_url
        self.auth = (user_name, password)
        self.logger = logger


    def get_request(self, type, period, from_timestamp):
        return requests.get(
            f"{self.base_url}/{type}/{period}/{from_timestamp}",
            auth=self.auth)


    def get_ohlcv(self, period, from_timestamp):
        return self.get_request('ohlcv', period, from_timestamp)


    def get_close(self, period, from_timestamp):
        return self.get_request('close', period, from_timestamp)
