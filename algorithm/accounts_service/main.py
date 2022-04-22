import os
import json
from abc import abstractmethod
import pandas as pd
import numpy as np
from libs import *
"""from flask import Flask, jsonify, make_response
from flask.wrappers import Response
from flask_httpauth import HTTPBasicAuth
from flask_apscheduler import APScheduler

class Flask_App_Config:
    
    SCHEDULER_API_ENABLED = True
"""


class Account():
    currencies = []
    bots = []

    def __init__(self, db=None, account_id=None):
        if not(db is None):
            self.init_from_db(db, account_id)
        

    @staticmethod
    def get_currencies_from_pair(pair):
        return pair.split('/')

    @staticmethod
    def get_currencies_from_pairs(pairs):
        result = []
        for pair in pairs:
            result += pair.split('/')
        return list(set(result))

    def init_from_db(self, db, account_id):
        self.account_id = account_id
        exchnage_id, self.bots_ids = db.get_account(account_id)
        self.exchange = Exchange(db, exchnage_id)
        self.bots = [Bot(db, bot_id) for bot_id in self.bots_ids]
        self.set_balances_zero(self.get_currencies_from_pairs(self.exchange.pairs))
        self.set_balances_from_db(db)


    def set_balances_zero(self, currencies):
        self.balances = {currency: 0.0 for currency in currencies}


    def set_balances_from_db(self, db):
        db_balances = db.get_last_balance_from_db(self.account_id)
        for currency, value in db_balances.items():
            self.balances[currency] = value


    def set_balance_from_exchange(self):
        pass


    def get_balance_current(self):
        pass


    def buy(self, pair, amount):
        currencies = self.get_currencies_from_pair(pair)
        # request OHLCV from data service

    def sell(self, amount):
        pass

    def make_operation(self, pair, type, amount):
        pass

    def add_balance(self, currency, amount):
        self.balances[currency] += amount

    def realloc_balances(self, amount):
        pass


rest_api_user = os.environ.get("REST_API_USER")
rest_api_password = os.environ.get("REST_API_PASSWORD")

db = MongoDB(os.environ.get("MONGO_USERNAME"), os.environ.get("MONGO_PASSWORD"))

account = Account()
account.init_from_db(db, os.environ.get("DB_ACCOUNT_ID"))

def main():
    pass

if __name__ == '__main__':
    main()