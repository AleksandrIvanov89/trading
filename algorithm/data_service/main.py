import os
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, make_response
from flask.wrappers import Response
from flask_httpauth import HTTPBasicAuth
from flask_apscheduler import APScheduler


class Flask_App_Config:
    
    SCHEDULER_API_ENABLED = True

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

    def __init__(self, exchange_name, symbol, history_period, cleanup_period):
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


def load_config():
    """
    Load params from config file
    """
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange") # name from ccxt library
        symbol = json_data.get("symbol") # format - BTC/USDT
        history_period = json_data.get("history_period")
        cleanup_period = json_data.get("cleanup_period")
        return exchange_name, symbol, history_period, cleanup_period


def df_to_json(df):
    """
    Convert pandas dataframe to API response
    """
    return Response(
        df.to_json(orient="records"),
        mimetype='application/json'
        )

auth = HTTPBasicAuth()
app = Flask(__name__)
exchange_name, symbol, history_period, cleanup_period = load_config()
exchange = Exchange(exchange_name, symbol, history_period, cleanup_period)

app.config.from_object(Flask_App_Config())
scheduler = APScheduler()
scheduler.init_app(app)

rest_api_user = os.environ.get("REST_API_USER")
rest_api_password = os.environ.get("REST_API_PASSWORD")


def initialize():
    exchange.load_initial_ohlcvs()
    scheduler.start()


@auth.get_password
def get_password(username):
    """
    Authorization
    :param username: user name
    """
    if username == rest_api_user:
        return rest_api_password
    return None


@auth.error_handler
def unauthorized():
    """
    Handler of unauthorized access
    """
    return make_response(
        jsonify({'error': 'Unauthorized access'}),
        403
        )


@app.errorhandler(404)
def not_found(error):
    """
    Handler of error 404
    """
    return make_response(
        jsonify({'error': 'Not found'}),
        404
        )


@app.route('/ohlcv/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_ohlcv(period, from_timestamp):
    """
    Handle OHLCV request by period and from timestamp

    :param period: timeframe - 1m, 1h, 1d...
    :param from_timestamp: timestamp of the first ohlcv to return
    """
    return df_to_json(
        exchange.get_ohlcv_from_timestamp(
            period,
            from_timestamp
            )
        )


@app.route('/close/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_close(period, from_timestamp):
    """
    Handle Close request by period and from timestamp

    :param period: timeframe - 1m, 1h, 1d...
    :param from_timestamp: timestamp of the first close to return
    """
    return df_to_json(
        exchange.get_close_from_timestamp(
            period,
            from_timestamp
            )
        )

@scheduler.task('interval', id='update_1m', seconds=1, max_instances=1)
def update_1m():
    """
    Schedule update of 1m OHLCV
    """
    exchange.update_tohlcv('1m')


@scheduler.task('interval', id='update_1h', minutes=1, max_instances=1)
def update_1h():
    """
    Schedule update of 1h OHLCV
    """
    exchange.update_tohlcv('1h')


@scheduler.task('interval', id='update_1d', hours=1, max_instances=1)
def update_1d():
    """
    Schedule update of 1d OHLCV
    """
    exchange.update_tohlcv('1d')


if __name__ == '__main__':
    initialize()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
        )
