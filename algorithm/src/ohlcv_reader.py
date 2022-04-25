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
    """
    Flask app scheduler config
    """
    
    SCHEDULER_API_ENABLED = True


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

db = MongoDB(
    os.environ.get("MONGO_USERNAME"),
    os.environ.get("MONGO_PASSWORD"),
    logger=logger
    )

exchanges = {
    exchange_i["name"]: Exchange(db, exchange_i['id'], logger) for exchange_i in db.get_active_exchanges()
    }

app.config.from_object(Flask_App_Config())
scheduler = APScheduler()
scheduler.init_app(app)

rest_api_user = os.environ.get("REST_API_USER")
rest_api_password = os.environ.get("REST_API_PASSWORD")


def initialize():
    """
    Load OHLCVs from db and exchange and start scheduler
    """
    for exchange_i in exchanges.values():
        exchange_i.set_periods_params(
            os.environ.get("HISTORY_PERIOD"),
            os.environ.get("CLEANUP_PERIOD")
            )
        exchange_i.load_initial_ohlcvs(db)
    scheduler.start()


@auth.get_password
def get_password(username):
    """
    Authorization
    :param username: user name
    """
    return rest_api_password if username == rest_api_user else None


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
    :parama error: error
    """
    return make_response(
        jsonify({'error': f'Not found {error}'}),
        404
        )


@app.route('/ohlcv/<string:exchange_name>/<string:symbol_1>/<string:symbol_2>/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_ohlcv(exchange_name, symbol_1, symbol_2, period, from_timestamp):
    """
    Handle OHLCV request by period and from timestamp

    :param exchange_name: exchange name
    :param symbol_1: first symbol in a pair
    :param symbol_2: second symbol in a pair
    :param period: timeframe - 1m, 1h, 1d...
    :param from_timestamp: timestamp of the first ohlcv to return
    """
    if exchange_name in exchanges.keys():
        return df_to_json(
            exchanges[exchange_name].get_ohlcv_from_timestamp(
                Exchange.concat_pair(symbol_1, symbol_2),
                period,
                from_timestamp
                ))
    else:
        return df_to_json(pd.DataFrame([]))


@app.route('/close/<string:exchange_name>/<string:symbol_1>/<string:symbol_2>/<string:period>/<int:from_timestamp>', methods=['GET'])
@auth.login_required
def get_close(exchange_name, symbol_1, symbol_2, period, from_timestamp):
    """
    Handle Close request by period and from timestamp
    
    :param exchange_name: exchange name
    :param symbol_1: first symbol in a pair
    :param symbol_2: second symbol in a pair
    :param period: timeframe - 1m, 1h, 1d...
    :param from_timestamp: timestamp of the first close to return
    """
    if exchange_name in exchanges.keys():
        return df_to_json(
            exchanges[exchange_name].get_close_from_timestamp(
                Exchange.concat_pair(symbol_1, symbol_2),
                period,
                from_timestamp
                ))
    else:
        return df_to_json(pd.DataFrame([]))


@app.route('/current_close/<string:exchange_name>/<string:symbol_1>/<string:symbol_2>/<string:period>', methods=['GET'])
@auth.login_required
def get_current_close(exchange_name, symbol_1, symbol_2, period):
    """
    Handle Close request by period and from timestamp
    
    :param exchange_name: exchange name
    :param symbol_1: first symbol in a pair
    :param symbol_2: second symbol in a pair
    :param period: timeframe - 1m, 1h, 1d...
    """
    if exchange_name in exchanges.keys():
        return df_to_json(
            exchanges[exchange_name].get_last_close(
                Exchange.concat_pair(symbol_1, symbol_2),
                period
                ))
    else:
        return df_to_json(pd.DataFrame([]))


def update_exchanges(period):
    """
    Update OHLCVs for all exchanges
    :param period: timeframe - 1m, 1h, 1d...
    """
    for exchange_i in exchanges.values():
        exchange_i.check_update_all_pairs(period)


@scheduler.task('interval', id='update_1m', seconds=1, max_instances=1)
def update_1m():
    """
    Schedule update of 1m OHLCV
    """
    update_exchanges('1m')


@scheduler.task('interval', id='update_1h', minutes=1, max_instances=1)
def update_1h():
    """
    Schedule update of 1h OHLCV
    """
    update_exchanges('1h')


@scheduler.task('interval', id='update_1d', hours=1, max_instances=1)
def update_1d():
    """
    Schedule update of 1d OHLCV
    """
    update_exchanges('1d')


if __name__ == '__main__':
    initialize()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
        )
