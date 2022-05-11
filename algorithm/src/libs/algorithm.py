import os
import numpy as np
from .logger import *
from .bot import *
from .mongo import *
from .data_service_api import *

class Algorithm():
    
    def __init__(self, db, bot_id, logger):
        self.bot= Bot(db, bot_id)
        self.account_data_service_api = DataServiceAPI(
            os.environ.get("ACCOUNTS_REST_API_BASE_URL"),
            os.environ.get("REST_API_USER"),
            os.environ.get("REST_API_PASSWORD"),
            logger
            )
        self.logger = logger
        self.algorithm_id = self.bot.algorithm_id
        self.set_params(db.get_algorithm(self.algorithm_id))
        self.last_read_timestamp = 0
        self.initialize()


    def process(self):
        if self.get_data_from_exchange():
            self.process_data()


    def make_operation(self, operation_type, amount=None):
        self.account_data_service_api.post_operation(
            operation_type,
            self.bot.bot_id,
            amount
            )


    @abstractmethod
    def set_params(self, params):
        pass

    
    @abstractmethod
    def initialize(self):
        pass


    @abstractmethod
    def get_data_from_exchange(self):
        pass


    @abstractmethod
    def process_data(self):
        pass



