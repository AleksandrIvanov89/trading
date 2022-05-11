import pandas as pd
from datetime import datetime
import ccxt
from .logger import *
from .database import *
from .mongo import *

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

    price_types = ['ask', 'bid']
    
    state_run = False

    pairs = []
    ccxt_id = ''
    exchange_name = ''
    fee = 0.002 #!INIT FROM EXCHANGE

    def __init__(self, db=None, exchange_id=None, logger=None):
        if not(db is None):
            self.init_from_db(db, exchange_id)
        self.logger = logger
        

    def init_from_db(self, db, exchange_id):
        self.exchange_id = exchange_id
        self.exchange_name, self.ccxt_id, self.pairs = db.get_exchange(exchange_id)
        self.connect_to_exchange()


    def set_periods_params(self, history_period, cleanup_period):
        self.history_period = int(history_period)
        self.cleanup_period = int(cleanup_period)

    def connect_to_exchange(self):
        self.exchange = getattr(
            ccxt,
            self.ccxt_id
            )({'enableRateLimit': True, })
        self.markets = self.exchange.load_markets()


    def get_current_exchange_timestamp(self):
        return self.exchange.milliseconds()


    def calc_from_timestamp(self):
        """
        Calc timestamp for initial load of ohlcvs from exchange
        """
        return self.exchange.milliseconds() - self.periods['1m'] * self.history_period


    def load_ohlcv_from_exchange(self, pair, period, from_timestamp):
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
                    pair,
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
                log(
                    f"Exception in Exchange:{inspect.stack()[0][3]}\n{e}",
                    'exception',
                    self.logger
                    )

        cur_timestamp = self.exchange.milliseconds()
        cur_timestamp_cut = cur_timestamp - (cur_timestamp % self.periods[period])
        
        result = pd.DataFrame(tohlcv_list, columns=self.tohlcv_columns)
        
        result = result.loc[result['timestamp'] < cur_timestamp_cut]
        return result


    def load_initial_ohlcvs(self, db):
        """
        Load initial OHLCVs for all periods
        """
        from_timestamp = self.calc_from_timestamp()
        self.tohlcv = {}
        for pair in self.pairs:
            self.tohlcv[pair] = {}
            for period in self.periods.keys():
                try:
                    temp = self.tohlcv[pair][period] = db.get_ohlcv(
                        self.exchange_id,
                        pair,
                        period,
                        from_timestamp
                        )
                    print(temp)
                    if temp.shape[0] > 0:
                        self.tohlcv[pair][period] = temp[self.tohlcv_columns]
                        self.update_tohlcv(pair, period)
                    else:
                        self.tohlcv[pair][period] = self.load_ohlcv_from_exchange(
                            pair,
                            period,
                            from_timestamp
                            )
                except Exception as e:
                    log(
                        f"Exception in Exchange:{inspect.stack()[0][3]}\n{e}",
                        'exception',
                        self.logger
                        )
                    self.tohlcv[pair][period] = self.load_ohlcv_from_exchange(
                        pair,
                        period,
                        from_timestamp
                        )
                    
        self.state_run = True


    def check_update(self, pair, period):
        if (self.tohlcv[pair][period].shape[0] > 0) and self.state_run:
            self.update_tohlcv(pair, period)

    
    def check_update_all_pairs(self, period):
        if self.state_run:
            for pair in self.pairs:
                if self.tohlcv[pair][period].shape[0] > 0:
                    self.update_tohlcv(pair, period)
                else:
                    self.load_initial_ohlcvs(pair)


    def update_tohlcv(self, pair, period):
        """
        Get new OHLCVs from exchange

        :param period: timeframe - 1m, 1h, 1d...
        """
        last_timestamp = self.get_last_timestamp_from_df(pair, period)
        cur_timestamp = self.exchange.milliseconds()
        cur_timestamp_cut = cur_timestamp - (cur_timestamp % self.periods[period])
        if cur_timestamp_cut > last_timestamp + self.periods[period]:
            tohlcv_new = self.load_ohlcv_from_exchange(
                pair,
                period,
                last_timestamp + 1
                )
            if tohlcv_new.shape[0] > 0:
                self.tohlcv[pair][period] = pd.concat(
                    [self.tohlcv[pair][period], tohlcv_new],
                    ignore_index=True)
                self.tohlcv_cleanup(pair, period)

        
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
        return self.timestamp_to_str(self.exchange.milliseconds())


    @staticmethod
    def concat_pair(symbol_1, symbol_2):
        return ''.join((symbol_1, '/', symbol_2))


    def tohlcv_cleanup(self, pair, period):
        """
        Decrease the length of the buffer (self.tohlcv)
        """
        df_len = self.tohlcv[pair][period].shape[0]
        if df_len - self.cleanup_period > self.history_period:
            self.tohlcv[pair][period].drop(
                self.tohlcv[pair][period].index[0:df_len-self.history_period],
                inplace=True
                )


    def get_last_timestamp_from_df(self, pair, period):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        """
        return self.tohlcv[pair][period]['timestamp'].iat[-1]
                    

    def get_ohlcv_from_timestamp(self, pair, period, from_timestamp):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        :param from_timestamp: timestamp of the first OHLCV to return
        """
        print(f"Get OHLCV {period} from API starting from {from_timestamp}")
        if self.state_run and (pair in self.pairs) and (period in self.periods):
            return self.tohlcv[pair][period].loc[
                self.tohlcv[pair][period]['timestamp'] > from_timestamp
                ]
        else:
            return pd.DataFrame([])
    

    def get_close_from_timestamp(self, pair, period, from_timestamp):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        :param from_timestamp: timestamp of the first close to return
        """
        print(f"Get close {period} from API starting from {from_timestamp}")
        if self.state_run and (pair in self.pairs) and (period in self.periods):
            return self.tohlcv[pair][period][['timestamp', 'close']].loc[
                self.tohlcv[pair][period]['timestamp'] > from_timestamp
                ]
        else:
            return pd.DataFrame([])

    
    def get_last_close(self, pair, period):
        """
        Get last timestamp in self.tohlcv for period

        :param period: timeframe - 1m, 1h, 1d...
        """
        if self.state_run and (pair in self.pairs) and (period in self.periods):
            return self.tohlcv[pair][period][['timestamp', 'close']].loc[
                self.tohlcv[pair][period]['timestamp'].idxmax()
                ]
        else:
            return pd.DataFrame([])

    
    def get_ticker(self):
        ticker = {price_type: 0.0 for price_type in self.price_types}
        try:
            temp_ticker = self.exchange.fetch_ticker('BTC/USD')
            ticker = {'ask': temp_ticker['ask'], 'bid': temp_ticker['bid']}
        except Exception as e:
            log(
                f"Exception in Exchange:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )
        return ticker

    def calc_price_by_order_book(self, order_book, amount=1.0):
        if amount <= 0.0:
            amount = 1.0
        def calc_price_type(price_type):
            res = 0
            lim = amount
            i = 0
            while lim > 0:
                temp = min(order_book[price_type][i][1], lim)
                res += order_book[price_type][i][0] * temp
                lim -= temp
                i += 1
            return res / amount
        return {price_type: calc_price_type(price_type) for price_type in self.price_types}


    def get_price_from_order_book(self, amount):
        ticker = {price_type: 0.0 for price_type in self.price_types}
        try:
            order_book = self.exchange.fetch_order_book("BTC/USD")
            ticker = self.calc_price_by_order_book(order_book, amount)
        except Exception as e:
            log(
                f"Exception in Exchange:{inspect.stack()[0][3]}\n{e}",
                'exception',
                self.logger
                )
        return ticker

    
    def get_price(self, amount):
        ticker = self.get_price_from_order_book(amount)
        return ticker if all(price > 0.0 for price in ticker.values()) else self.get_ticker()


    def average_ticker(self, ticker):
        return (ticker['ask'] + ticker['bid']) / 2.0