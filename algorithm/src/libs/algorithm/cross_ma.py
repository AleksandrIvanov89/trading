import numpy as np
from .algorithm import *
from .ohlcv_algo import *

class Cross_MA(OHLCV_Algorithm):
    
    def __init__(self, db, bot_id, logger):
        super().__init__(db, bot_id, logger)

    
    def set_params(self, params):
        self.short_window = params['short_window']
        self.long_window = params['long_window']
        self.mode = params['mode']

    
    def initialize(self):
        self.initialize_ohlcv_from_period(
            self.long_window + self.mode + 1
            )
        self.ma_long = self.moving_average(
            self.timeseries[1,:],
            self.long_window
            )
        self.ma_short = self.moving_average(
            self.timeseries[1,:],
            self.short_window
            )
        self.ma_long_len = self.ma_long.shape[0]
        self.ma_short_len = self.ma_short.shape[0]
        self.prev_cross_temp = self.calc_cross()


    def moving_average(self, np_array, window_size):
        return np.convolve(
            np_array,
            np.ones(window_size), 'valid') / window_size


    def calc_new_ma(self, ma_array, window_size):
        return np.concatenate((
            ma_array,
            [np.average(self.timeseries[1, -window_size:])]
            ))
    

    def calc_cross(self):
        def filter_func(item):
            return 0 if item == 0 else item / abs(item)
        return np.sum(np.stack(np.vectorize(filter_func)(
            self.ma_short[-self.mode:]-self.ma_long[-self.mode:]
            )))
    

    def process_data(self):
        for item in self.new_ohlcv:
            self.timeseries = np.concatenate(
                (
                    self.timeseries,
                    np.array([[item['timestamp']],[item['close']]])
                    ),
                axis=1
                )

            self.ma_long = self.calc_new_ma(
                self.ma_long,
                self.long_window
                )
            self.ma_short = self.calc_new_ma(
                self.ma_short,
                self.short_window
                )
        new_cross = self.calc_cross()

        if (self.prev_cross_temp > 0) and (new_cross < 0):
            self.make_operation('sell_all')
        elif (self.prev_cross_temp < 0) and (new_cross > 0):
            self.make_operation('buy_all')

        self.prev_cross_temp = new_cross
        self.ma_long = self.ma_long[-self.ma_long_len:]
        self.ma_short = self.ma_short[-self.ma_short_len:]
        self.timeseries = self.timeseries[:,-(self.long_window + self.mode + 1):]


    