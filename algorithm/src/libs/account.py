from .logger import *
from .exchange import *
from .mongo import *
from .tradingbot import TradingBot

class Account():

    symbols = []
    bots = {}

    def __init__(self, db=None, account_id=None, logger=None):
        self.logger = logger
        if not(db is None):
            self.init_from_db(db, account_id)


    @staticmethod
    def get_symbols_from_pair(pair):
        return pair.split('/')


    @staticmethod
    def get_symbols_from_pairs(pairs):
        result = []
        for pair in pairs:
            result += pair.split('/')
        return list(set(result))


    def init_from_db(self, db, account_id):
        self.account_id = account_id
        self.exchange_id, bots_ids, self.type = db.get_account(account_id)
        self.exchange = Exchange(db, self.exchange_id, self.logger)
        self.bots = {bot_id: TradingBot(db, bot_id) for bot_id in bots_ids}
        self.set_balances_zero(self.get_symbols_from_pairs(self.exchange.pairs))
        self.set_balances_from_db(db)


    def set_balances_zero(self, symbols):
        self.balances = {symbol: 0.0 for symbol in symbols}


    def set_balances_from_db(self, db):
        db_balances = db.get_account_last_balance(self.account_id)
        for symbol, value in db_balances.items():
            self.balances[symbol] = value


    def set_balance_from_exchange(self):
        pass


    def get_balance_current(self):
        res = self.balances.copy()
        res.update({'timestamp': self.exchange.get_current_exchange_timestamp()})
        return res


    def get_balance_current_by_symbol(self, symbol):
        return {
            'timestamp': self.exchange.get_current_exchange_timestamp(),
            symbol: self.balances[symbol]
            }


    def apply_bot_operation(self, diff):
        for symbol in diff.keys():
            self.balances[symbol] += diff[symbol]
    

    def make_operation(self, operation_type, bot_id, amount=None):
        if bot_id in self.bots.keys():
            self.apply_bot_operation(
                self.bots[bot_id].make_operation(operation_type, amount)
                )


    def change_balance(self, symbol, amount):
        self.balances[symbol] = min(
            sum(bot.get_balance(symbol) for bot in self.bots),
            self.balances[symbol] + amount
            )


    def change_bot_balance(self, bot_id, symbol, amount):
        if bot_id in self.bots.keys():
            if symbol in self.bots[bot_id].symbols:
                self.bots[bot_id].balance[symbol] = max(
                    0.0,
                    self.bots[bot_id].balance[symbol] + amount
                    )

