import os
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, make_response
from flask.wrappers import Response
from flask_httpauth import HTTPBasicAuth
from flask_apscheduler import APScheduler
from libs import *

class Flask_App_Config:
    
    SCHEDULER_API_ENABLED = True


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

logger = Logger("/logs/logs.log")

db = MongoDB(os.environ.get("MONGO_USERNAME"), os.environ.get("MONGO_PASSWORD"))
exchange_name, symbol, history_period, cleanup_period = load_config()

exchange = Exchange(logger, db, os.environ.get("EXCHANGE_ID"))
exchange.set_periods_params(os.environ.get("HISTORY_PERIOD"), os.environ.get("CLEANUP_PERIOD"))
#db = MongoDB(exchange_name, symbol, os.environ.get("MONGO_USERNAME"), os.environ.get("MONGO_PASSWORD"))


app.config.from_object(Flask_App_Config())
scheduler = APScheduler()
scheduler.init_app(app)

rest_api_user = os.environ.get("REST_API_USER")
rest_api_password = os.environ.get("REST_API_PASSWORD")


def initialize():
    exchange.load_initial_ohlcvs(db)
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
            ) if period in exchange.periods.keys() else pd.DataFrame([])
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
            ) if period in exchange.periods.keys() else pd.DataFrame([])
        )
    

@scheduler.task('interval', id='update_1m', seconds=1, max_instances=1)
def update_1m():
    """
    Schedule update of 1m OHLCV
    """
    exchange.check_update('1m')


@scheduler.task('interval', id='update_1h', minutes=1, max_instances=1)
def update_1h():
    """
    Schedule update of 1h OHLCV
    """
    exchange.check_update('1h')


@scheduler.task('interval', id='update_1d', hours=1, max_instances=1)
def update_1d():
    """
    Schedule update of 1d OHLCV
    """
    exchange.check_update('1d')


if __name__ == '__main__':
    initialize()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
        )
