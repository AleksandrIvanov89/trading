import os
from .algorithm import *
from .data_service_api import *

class OHLCV_Algorithm(Algorithm):

    def __init__(self, db, bot_id, logger):
        self.ohlcv_data_service_api = DataServiceAPI(
            os.environ.get("OHLCV_REST_API_BASE_URL"),
            os.environ.get("REST_API_USER"),
            os.environ.get("REST_API_PASSWORD"),
            logger
            )
        super().__init__(db, bot_id, logger)
        


    def np_from_response(self, response):
        """print(response)
        return np.fromiter(
            response.items(),
            dtype=np.float32,
            count=len(response)
            )"""
        return np.array([[item['timestamp'] for item in response],[item['close'] for item in response]])


    def initialize_ohlcv_from_period(self, period):
        self.bot.exchange.set_periods_params(
            period,
            int(period / 10)
            )
        result = self.ohlcv_data_service_api.get_close(
            self.bot.exchange.exchange_name,
            self.bot.pair,
            '1m',
            self.bot.exchange.calc_from_timestamp()
            )
        self.timeseries = self.np_from_response(result)
        self.last_read_timestamp = self.timeseries[0, -1]


    def get_data_from_exchange(self):
        cur_timestamp = self.bot.exchange.get_current_exchange_timestamp()
        if cur_timestamp - self.last_read_timestamp >= self.bot.exchange.periods['1m']:
            self.new_ohlcv = self.ohlcv_data_service_api.get_close(
                self.bot.exchange.exchange_name,
                self.bot.pair,
                '1m',
                self.last_read_timestamp
                )
            if len(self.new_ohlcv) > 0:
                self.last_read_timestamp = self.new_ohlcv[-1]['timestamp']
                return True
        return False

        