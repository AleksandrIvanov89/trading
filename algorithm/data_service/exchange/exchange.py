import pandas as pd
from datetime import datetime
import ccxt
from .logger import *

class Exchange():
    
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
    
    state_run = False

    def __init__(self, exchange_name, symbol, history_period, cleanup_period, logger=None):
        self.logger = logger
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.history_period = history_period
        self.cleanup_period = cleanup_period
        self.exchange = getattr(
            ccxt,
            self.exchange_name)(
                {
                    'enableRateLimit': True, 
                })
        self.markets = self.exchange.load_markets()


    def calc_from_timestamp(self):
        """
        Calc timestamp for initial load of ohlcvs from exchange
        """
        return self.exchange.milliseconds() - self.periods['1m'] * self.history_period


    def load_ohlcv_from_exchange(self, period, from_timestamp):
        """
        Load OHLCVs from exchange

        :param period: timeframe - 1m, 1h, 1d...
        :param from_timestamp: timestamp of the first OHLCV to collect from exchange
        """
        prev_from_timestamp = 0
        tohlcv_list = []
        
        while prev_from_timestamp != from_timestamp:
            try:
                print(f"Loading OHLCVs starting from {from_timestamp}")
                tohlcv_list_temp = self.exchange.fetch_ohlcv(
                    self.symbol,
                    period,
                    from_timestamp)
                # append data
                if len(tohlcv_list_temp) > 0:
                    if len(tohlcv_list) > 0:
                        tohlcv_list += tohlcv_list_temp
                    else:
                        tohlcv_list = tohlcv_list_temp
                # loop variables
                prev_from_timestamp = from_timestamp
                if len(tohlcv_list) > 0:
                    from_timestamp = tohlcv_list[-1][0] + 1
            except Exception as e:
                if self.logger != None:
                    self.logger.exception(e)
                else:
                    print(f"Error: {e}")

        cur_timestamp = self.exchange.milliseconds()
        cur_timestamp_cut = cur_timestamp - (cur_timestamp % self.periods[period])
        
        result = pd.DataFrame(tohlcv_list, columns=self.tohlcv_columns)
        
        result = result.loc[result['timestamp'] < cur_timestamp_cut]
        return result

    def load_initial_ohlcvs(self):
        """
        Load initial OHLCVs for all periods
        """
        if self.logger != None:
            self.logger.info(f"Loading initial OHLCVs")
        else:
            print(f"Loading initial OHLCVs")
        from_timestamp = self.calc_from_timestamp()
        self.tohlcv = {}
        for period in self.periods.keys():
            self.tohlcv[period] = self.load_ohlcv_from_exchange(period, from_timestamp)
        self.state_run = True


    def update_tohlcv(self, period):
        """
        Get new OHLCVs from exchange

        :param period: timeframe - 1m, 1h, 1d...
        """
        if (self.tohlcv[period].shape[0] > 0) and self.state_run:
            last_timestamp = self.get_last_timestamp_from_df(period)
            cur_timestamp = self.exchange.milliseconds()
            cur_timestamp_cut = cur_timestamp - (cur_timestamp % self.periods[period])
            if cur_timestamp_cut > last_timestamp + self.periods[period]:
                tohlcv_new = self.load_ohlcv_from_exchange(period, last_timestamp + 1)
                if tohlcv_new.shape[0] > 0:
                    self.tohlcv[period] = pd.concat(
                        [
                            self.tohlcv[period],
                            tohlcv_new
                            ],
                        ignore_index=True
                        )
                    self.tohlcv_cleanup(period)

        
    @staticmethod
    def timestamp_to_str(timestamp):
        """
        Convert the timestamp to a string representation

        :param timestamp: timestamp to convert to a string representation
        """
        return datetime.fromtimestamp(int(timestamp/1000)).strftime("%m/%d/%Y, %H:%M:%S")

    
    def exchange_time_str(self):
        """
        Convert the current exchange time to a string representation
        """
        return self.timestamp_to_str(
            self.exchange.milliseconds()
            )


    def tohlcv_cleanup(self, period):
        """
        Decrease the length of the buffer (self.tohlcv)
        """
        df_len = self.tohlcv[period].shape[0]
        if df_len - self.cleanup_period > self.history_period:
            self.tohlcv[period].drop(self.tohlcv[period].index[0:df_len-self.history_period], inplace=True)


    def get_last_timestamp_from_df(self, period):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        """
        return self.tohlcv[period]['timestamp'].iat[-1]
                    

    def get_ohlcv_from_timestamp(self, period, from_timestamp):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        :param from_timestamp: timestamp of the first OHLCV to return
        """
        print(f"Get OHLCV {period} from API starting from {from_timestamp}")
        if self.state_run:
            return self.tohlcv[period].loc[self.tohlcv[period]['timestamp'] > from_timestamp]
        else:
            return pd.DataFrame([])
    

    def get_close_from_timestamp(self, period, from_timestamp):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        :param from_timestamp: timestamp of the first close to return
        """
        print(f"Get close {period} from API starting from {from_timestamp}")
        if self.state_run:
            return self.tohlcv[period][['timestamp', 'close']].loc[self.tohlcv[period]['timestamp'] > from_timestamp]
        else:
            return pd.DataFrame([])