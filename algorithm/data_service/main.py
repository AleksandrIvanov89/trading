import os
import json
import ccxt
import pandas as pd
import numpy as np

from flask import Flask, jsonify, abort, make_response, request
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
        print(f"{self.exchange_name} exchange successfully initialized")


    def calc_from_timestamp(self):
        return self.exchange.milliseconds() - self.periods['1m'] * self.history_period


    def load_ohlcv(self, period, from_timestamp):
        print(f"Load from exchange {self.exchange_name} OHLCV {period} starting from {from_timestamp}")
        prev_from_timestamp = 0
        tohlcv_list = []
        while prev_from_timestamp != from_timestamp:
            try:
                print(f"Requesting data starting from {from_timestamp}")
                tohlcv_list_temp = self.exchange.fetch_ohlcv(
                    self.symbol,
                    period,
                    from_timestamp)
                print(f"Received {len(tohlcv_list_temp)} OHLCVs, Accumulated OHLCVs {len(tohlcv_list)}")
                # append data
                if len(tohlcv_list_temp) > 0:
                    if len(tohlcv_list) > 0:
                        if tohlcv_list_temp[-1][0] != tohlcv_list[-1][0]:
                            tohlcv_list += tohlcv_list_temp
                    else:
                        tohlcv_list = tohlcv_list_temp
                print(f"OHLCVs {len(tohlcv_list)} after append")
                # loop variables
                prev_from_timestamp = from_timestamp
                if len(tohlcv_list) > 0:
                    from_timestamp = tohlcv_list[-1][0]
            except Exception as e:
                print(f"Error: {e}")
                print(f"Args: {e.args}")
        return tohlcv_list
        

    def init_ohlcv(self):
        from_timestamp = self.calc_from_timestamp()
        print(f"Init OHLCVs starting from {from_timestamp}")
        self.tohlcv = {}
        for period in self.periods.keys():
            print(f"Init OHLCVs starting from {from_timestamp} for period {period}")
            tohlcv_list = self.load_ohlcv(period, from_timestamp)
            self.tohlcv[period] = pd.DataFrame(
                tohlcv_list,
                columns=self.tohlcv_columns)
            self.tohlcv[period].drop_duplicates(
                self.tohlcv_columns[0],
                inplace=True,
                keep='last')
            print(f"After init df {period} contains {self.tohlcv[period].shape[0]}")
        print(f"All dfs initialized")
        self.state_run = True


    def tohlcv_cleanup(self, period):
        df_len = self.tohlcv[period].shape[0]
        print(f"Need cleanup df {period}")
        if df_len - self.cleanup_period > self.history_period:
            print(f"Cleanup df {period}")
            self.tohlcv[period].drop(
                self.tohlcv[period].index[0:df_len-self.history_period],
                inplace=True)


    def get_last_timestamp_from_df(self, period):
        return self.tohlcv[period]['timestamp'].iat[-1]

    
    def check_need_to_update(self, period, last_timestamp):
        #print(f"Checking for {period}: {self.exchange.milliseconds()} - {self.periods[period]} >= {last_timestamp}")
        cur_timestamp = self.exchange.milliseconds()
        return cur_timestamp - (cur_timestamp % self.periods[period]) - self.periods[period] >= last_timestamp


    def update_tohlcv(self, period):
        if self.tohlcv[period].shape[0] > 0:
            last_timestamp = self.get_last_timestamp_from_df(period)
            if self.check_need_to_update(period, last_timestamp):
                print(f"Update started {ccxt.Exchange.iso8601(self.exchange.milliseconds())} for period {period}")
                print(f"Last timestamp added {last_timestamp} - {ccxt.Exchange.iso8601(int(last_timestamp/1000))} for period {period}")
                tohlcv_list = self.load_ohlcv(period, last_timestamp)
                if len(tohlcv_list) > 0:# check the last elem
                    print(f"OHLCV {period} loaded {len(tohlcv_list)}\n{tohlcv_list}")
                    print(f"OHLCV before update {self.tohlcv[period].tail(5)}")
                    self.tohlcv[period] = pd.concat(
                        [
                            self.tohlcv[period],
                            pd.DataFrame(tohlcv_list, columns=self.tohlcv_columns)
                        ],
                        ignore_index=True
                        )
                    self.tohlcv[period].drop_duplicates(
                        self.tohlcv_columns[0],
                        inplace=True,
                        keep='last'
                        )
                    self.tohlcv_cleanup(period)
                    print(f"OHLCV after update {self.tohlcv[period].tail(5)}")
                    
                    


    def update(self):
        if self.state_run:
            for period in self.periods.keys():
                self.update_tohlcv(period)


    def get_ohlcv_from_timestamp(self, period, from_timestamp):
        print(f"Get OHLCV {period} from API starting from {from_timestamp}")
        if self.state_run:
            return self.tohlcv[period].loc[self.tohlcv[period]['timestamp'] > from_timestamp]
        else:
            return pd.DataFrame([])
    

    def get_close_from_timestamp(self, period, from_timestamp):
        print(f"Get close {period} from API starting from {from_timestamp}")
        if self.state_run:
            return self.tohlcv[period][['timestamp', 'close']].loc[self.tohlcv[period]['timestamp'] > from_timestamp]
        else:
            return pd.DataFrame([])


def load_config():
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange") # name from ccxt library
        symbol = json_data.get("symbol") # format - BTC/USDT
        history_period = json_data.get("history_period")
        cleanup_period = json_data.get("cleanup_period")
        return exchange_name, symbol, history_period, cleanup_period


def df_to_json(df):
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


@auth.get_password
def get_password(username):
    if username == rest_api_user:
        return rest_api_password
    return None


@auth.error_handler
def unauthorized():
    return make_response(
        jsonify({'error': 'Unauthorized access'}),
        403
        )


@app.errorhandler(404)
def not_found(error):
    return make_response(
        jsonify({'error': 'Not found'}),
        404
        )


@app.route('/ohlcv/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_ohlcv(period, from_timestamp):
    return df_to_json(
        exchange.get_ohlcv_from_timestamp(
            period,
            from_timestamp
            )
        )


@app.route('/close/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_close(period, from_timestamp):
    return df_to_json(
        exchange.get_close_from_timestamp(
            period,
            from_timestamp
            )
        )

@scheduler.task('interval', id='update', seconds=1, max_instances=1)
def update():
    if exchange.state_run:
        exchange.update()
    else:
        exchange.init_ohlcv()


if __name__ == '__main__':
    scheduler.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
