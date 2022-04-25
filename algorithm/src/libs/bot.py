from .exchange import *
from .account import *

class Bot():
    
    bot_id = ''
    balance = {}#{'BTC': 1.0, 'USDT': 1.0}
    symbols = []
    order_types = ['buy', 'sell', 'buy_all', 'sell_all']

    def __init__(self, db=None, bot_id=None):
        if not(db is None):
            self.init_from_db(db, bot_id)

    def init_from_db(self, db, bot_id):
        self.bot_id = bot_id
        self.exchange_id, self.pair, self.algorithm_id, self.type, self.state = db.get_bot(bot_id)
        self.symbols = Account.get_symbols_from_pair(self.pair)
        self.exchange = Exchange()
        self.exchange.init_from_db(db, self.exchange_id)

    def init_balances_from_db(self, db):
        pass

    def make_diff(self, before):
        return {symbol: self.balance[symbol] - before[symbol] for symbol in self.symbols}

    def get_balance(self, symbol):
        return self.balance[symbol] if symbol in self.symbols else 0.0

    def make_operation(self, operation_type, amount=None):
        before = self.balance.copy()
        if operation_type in self.order_types:
            getattr(self, operation_type)(amount) 
        return self.make_diff(before)
        

    def buy_all(self, amount):
        price = self.exchange.get_price(self.balance[self.symbols[1]] / self.exchange.get_ticker())
        self.balance[self.symbols[0]] += self.balance[self.symbols[1]] / price * (1.0 - self.exchange.fee)
        self.balance[self.symbols[1]] = 0.0
        


    def sell_all(self, amount):
        price = self.exchange.get_price(self.balance[self.symbols[0]])
        self.balance[self.symbols[1]] = self.balance[self.symbols[0]] * price
        self.balance[self.symbols[0]] = 0.0
        

    def buy(self, amount):
        if amount > 0:
            price = self.exchange.get_price(amount)
            corrected_amount = min(self.balance[self.symbols[1]] / price['ask'], amount)
            self.balance[self.symbols[0]] += corrected_amount * (1.0 - self.exchange.fee)
            self.balance[self.symbols[1]] -= corrected_amount * price['ask']
            

    def sell(self, amount):
        if amount > 0:
            price = self.exchange.get_price(amount)
            corrected_amount = min(self.balance[self.symbols[0]], amount)
            self.balance[self.symbols[0]] -= corrected_amount
            self.balance[self.symbols[1]] += corrected_amount * price['bid'] * (1.0 - self.exchange.fee)