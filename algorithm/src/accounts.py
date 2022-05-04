import os
import json
from libs import *
from flask import Flask, jsonify, make_response, request
from flask.wrappers import Response
from flask_httpauth import HTTPBasicAuth

#from flask_apscheduler import APScheduler

"""class Flask_App_Config: # ! Add balance synchronization with exchange
    
    SCHEDULER_API_ENABLED = True"""


auth = HTTPBasicAuth()
app = Flask(__name__)

logger = Logger("/logs/logs.log")

rest_api_user = os.environ.get("REST_API_USER")
rest_api_password = os.environ.get("REST_API_PASSWORD")

db = MongoDB(
    os.environ.get("MONGO_USERNAME"),
    os.environ.get("MONGO_PASSWORD"),
    "mongodb:27017",
    logger
    )

accounts = {
    account['_id']: Account(db, account['_id'], logger)\
        for account in db.get_all_accounts()
    }

bots = {}

"""app.config.from_object(Flask_App_Config())
scheduler = APScheduler()
scheduler.init_app(app)"""

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


@app.route('/all_accounts_balances', methods=['GET'])
@auth.login_required
def get_all_accounts_balances():
    return Response(
        json.dumps([account.get_balance_current()\
            for account in accounts.values()]),
        mimetype='application/json'
        )


@app.route('/all_bots_balances', methods=['GET'])
@auth.login_required
def get_all_bots_balances():
    res = []
    for account in accounts.values():
        for bot in account.bots.values():
            res.append(bot.get_balances())
    return Response(json.dumps(res), mimetype='application/json')


@app.route('/account_balances/<string:account_id>', methods=['GET'])
@auth.login_required
def get_account_balances(account_id):
    return Response(
        json.dumps(accounts[account_id].get_balance_current()\
            if account_id in accounts.keys() else {}),
        mimetype='application/json'
        )


@app.route('/bot_balances/<string:bot_id>', methods=['GET'])
@auth.login_required
def get_bot_balances(bot_id):
    return Response(
        json.dumps(accounts[bots[bot_id]].bots[bot_id].get_balances()\
            if bot_id in bots.keys() else {}),
        mimetype='application/json'
        )


@app.route(
    '/make_operation',
    methods=['POST']
    )
@auth.login_required
def make_operation():
    bot_id = request.form['bot_id']
    account_id = bots[bot_id]
    if accounts[account_id].make_operation(
        request.form['operation_type'],
        bot_id,
        request.form['amount']
        ):
        db.write_operation(
            request.form['operation_type'],
            account_id,
            bot_id,
            request.form['amount'],
            accounts[account_id].bots[bot_id].pair,
            accounts[account_id].exchange.get_current_exchange_timestamp()
            )


def initialize():
    for account in accounts.values():
        for bot in account.bots.keys():
            bots.update({bot: account.account_id})
            

if __name__ == '__main__':
    initialize()
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True,
        use_reloader=False
        )