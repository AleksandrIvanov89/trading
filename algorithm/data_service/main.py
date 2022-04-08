import os
import json
import ccxt
import pandas as pd
import numpy as np

from flask import Flask, jsonify, abort, make_response, request
from flask.wrappers import Response
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()
app = Flask(__name__)
rest_api_user = os.environ.get("REST_API_USER")
rest_api_password = os.environ.get("REST_API_PASSWORD")


class Exchange():

    periods = {
        '1m': 60000,
        '1h': 360000,
        '1d': 86400000
    }
    cleanup_period = 1000

    tohlcv_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    

    def __init__(self, exchange_name, symbol, history_period):
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.history_period = history_period
        self.exchange = getattr(
            ccxt,
            self.exchange_name)(
                {
                    'enableRateLimit': True, 
                })
        self.markets = self.exchange.load_markets()


    def calc_from_timestamp(self):
        return self.exchange.milliseconds() - self.periods['1m'] * self.history_period


    def load_ohlcv(self, period, from_timestamp):
        
        prev_from_timestamp = 0
        tohlcv_list = []
        while prev_from_timestamp != from_timestamp:
            try:
                tohlcv_list_temp = self.exchange.fetch_ohlcv(
                    self.symbol,
                    period,
                    from_timestamp)
                # append data
                if len(tohlcv_list_temp) > 0:
                    if len(tohlcv_list) > 0:
                        if tohlcv_list_temp[-1][0] != tohlcv_list[-1][0]:
                            tohlcv_list += tohlcv_list_temp
                    else:
                        tohlcv_list = tohlcv_list_temp
                # loop variables
                prev_from_timestamp = from_timestamp
                if len(tohlcv_list) > 0:
                    from_timestamp = tohlcv_list[-1][0]
            except Exception as e:
                print("Error: ", e)
        return tohlcv_list
        

    def init_ohlcv(self):
        from_timestamp = self.calc_from_timestamp()
        self.tohlcv = {}
        for period in self.periods.keys():
           tohlcv_list = self.load_ohlcv(period, from_timestamp)
           self.tohlcv[period] = pd.DataFrame(
               tohlcv_list,
               columns=self.tohlcv_columns)


    def update_tohlcv(self, period):
        last_timestamp = self.tohlcv[period]['timestamp'].iat[-1]
        if (last_timestamp <= self.exchange.milliseconds() - self.periods[period]):
            tohlcv_list = self.load_ohlcv(period, last_timestamp)
            if len(tohlcv_list) > 0:
                self.tohlcv[period] = pd.concat(
                    [
                        self.tohlcv[period],
                        pd.DataFrame(tohlcv_list, columns=self.tohlcv_columns)
                    ],
                    ignore_index=True)
                df_len = self.tohlcv[period].shape[0]
                if df_len - self.cleanup_period > self.history_period:
                    self.tohlcv[period].drop(
                        self.tohlcv[period].index[0:df_len-self.history_period],
                        inplace=True)


    def get_ohlcv_from_timestamp(self, period, from_timestamp):
        self.update_tohlcv(period)
        return self.tohlcv[period].loc[self.tohlcv[period]['timestamp'] > from_timestamp]
    

    def get_close_from_timestamp(self, period, from_timestamp):
        self.update_tohlcv(period)
        return self.tohlcv[period][['timestamp', 'close']].loc[self.tohlcv[period]['timestamp'] > from_timestamp]


def load_config():
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange") # name from ccxt library
        symbol = json_data.get("symbol") # format - BTC/USDT
        history_period = json_data.get("history_period")
        return exchange_name, symbol, history_period


def df_to_json(df):
    return Response(df.to_json(orient="records"), mimetype='application/json')


@auth.get_password
def get_password(username):
    if username == rest_api_user:
        return rest_api_password
    return None


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 403)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/ohlcv/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_ohlcv(period, from_timestamp):
    return df_to_json(exchange.get_ohlcv_from_timestamp(period, from_timestamp))


@app.route('/close/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_close(period, from_timestamp):
    return df_to_json(exchange.get_close_from_timestamp(period, from_timestamp))


if __name__ == '__main__':
    exchange_name, symbol, history_period = load_config()
    global exchange
    exchange = Exchange(exchange_name, symbol, history_period)
    exchange.init_ohlcv()
    app.run(host='0.0.0.0', port=5000, debug=True)
