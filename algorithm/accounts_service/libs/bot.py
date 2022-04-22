"""from .exchange import *"""

class Bot():
    
    bot_id = ''

    def __init__(self, db=None, bot_id=None):
        if not(db is None):
            self.init_from_db(db, bot_id)

    def init_from_db(self, db, bot_id):
        self.bot_id = bot_id
        self.exchange_id, self.symbol, self.algorithm_id, self.type, self.state = db.get_bot(bot_id)
        """self.exchange = Exchange()
        self.exchange.init_from_db(db, self.exchange_id)"""

    def init_balances_from_db(self, db):
        pass