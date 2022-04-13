import os
import json
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
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


    def calc_from_timestamp(self):
        return self.exchange.milliseconds() - self.periods['1m'] * self.history_period


    def load_ohlcv_from_exchange(self, period, from_timestamp):
        
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
        from_timestamp = self.calc_from_timestamp()
        self.tohlcv = {}
        for period in self.periods.keys():
            self.tohlcv[period] = self.load_ohlcv_from_exchange(period, from_timestamp)
        self.state_run = True


    def update_tohlcv(self, period):
        if (self.tohlcv[period].shape[0] > 0) and self.state_run:
            last_timestamp = self.get_last_timestamp_from_df(period)
            cur_timestamp = self.exchange.milliseconds()
            cur_timestamp_cut = cur_timestamp - (cur_timestamp % self.periods[period])
            print(f"Current exchange timestamp {cur_timestamp}")
            print(f"Current exchange time {self.exchange_time_str()}")
            print(f"Last in df timestamp {last_timestamp}")
            print(f"Last in df time {self.timestamp_to_str(last_timestamp)}")
            print(f"Last OHLCV for period {period}\n{self.tohlcv[period].tail(3)}")
            print("COMPARISON")
            print(f"{cur_timestamp_cut}\n{last_timestamp + self.periods[period]}")
            print(f"{cur_timestamp_cut - last_timestamp - self.periods[period]}")
            if cur_timestamp_cut > last_timestamp + self.periods[period]:
                tohlcv_new = self.load_ohlcv_from_exchange(period, last_timestamp + 1)
                if tohlcv_new.shape[0] > 0:
                    self.tohlcv[period] = pd.concat([self.tohlcv[period], tohlcv_new], ignore_index=True)
                    self.tohlcv_cleanup(period)
                    print(f"Last OHLCV after update for period {period}\n{self.tohlcv[period].tail(3)}")

        
    @staticmethod
    def timestamp_to_str(timestamp):
         return datetime.fromtimestamp(int(timestamp/1000)).strftime("%m/%d/%Y, %H:%M:%S")

    
    def exchange_time_str(self):
        return self.timestamp_to_str(self.exchange.milliseconds())


    def tohlcv_cleanup(self, period):
        df_len = self.tohlcv[period].shape[0]
        if df_len - self.cleanup_period > self.history_period:
            self.tohlcv[period].drop(self.tohlcv[period].index[0:df_len-self.history_period], inplace=True)


    def get_last_timestamp_from_df(self, period):
        return self.tohlcv[period]['timestamp'].iat[-1]
                    

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
exchange.load_initial_ohlcvs()

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

@scheduler.task('interval', id='update', seconds=10, max_instances=1)
def update():
    #if exchange.state_run:
    exchange.update()
    #else:
    #    exchange.load_initial_ohlcvs()


if __name__ == '__main__':
    scheduler.start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
