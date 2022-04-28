import os
import json
from libs import *

from flask import Flask, jsonify, make_response
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

"""data_service_api = DataServiceAPI(
    os.environ.get("REST_API_BASE_URL"),
    os.environ.get("REST_API_USER"),
    os.environ.get("REST_API_PASSWORD"),
    logger
    )"""

db = MongoDB(
    os.environ.get("MONGO_USERNAME"),
    os.environ.get("MONGO_PASSWORD"),
    "mongodb:27017",
    logger
    )

accounts = {
    account['_id']: Account(db, account['_id'], logger) for account in db.get_all_accounts()
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


@app.route('/account_balances/<string:account_id>', methods=['GET'])
@auth.login_required
def get_account_balances(account_id):
    return jsonify(
        accounts[account_id].get_balance_current() if account_id in accounts.keys() else {}
        )


@app.route('/bot_balances/<string:bot_id>', methods=['GET'])
@auth.login_required
def get_bot_balances(bot_id):
    return jsonify(
        accounts[bots[bot_id]].bots[bot_id].get_balances() if bot_id in bots.keys() else {}
        )


@app.route('/make_operation/<string:operation_type>/<string:bot_id>/<int:amount>', methods=['POST'])
@auth.login_required
def make_operation(operation_type, bot_id, amount):
    accounts[bots[bot_id]].make_operation(operation_type, bot_id, amount)


def initialize():
    print(accounts)
    #print(accounts.balances)
    for account in accounts.values():
        for bot in account.bots.keys():
            bots.update({bot: account.account_id})
            

if __name__ == '__main__':
    initialize()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
        )