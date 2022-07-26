import os
import json
import time
import schedule

from libs import *

logger = Logger("/logs/logs.log")
    
data_service_api = DataServiceAPI(
    os.environ.get("OHLCV_REST_API_BASE_URL"),
    os.environ.get("REST_API_USER"),
    os.environ.get("REST_API_PASSWORD"),
    logger
    )

db = MongoDB(
    os.environ.get("MONGO_USERNAME"),
    os.environ.get("MONGO_PASSWORD"),
    "mongodb:27017"
    )

exchanges = {
    exchange_i["name"]: Exchange(db, exchange_i['id'], logger)\
        for exchange_i in db.get_active_exchanges()
    }

def update(exchange, pair, period):
    last_timestamp_from_db = db.get_last_timestamp(
        exchange.exchange_id,
        pair,
        period
        )
    tohlcv_list = data_service_api.get_ohlcv(
        exchange.exchange_name,
        pair,
        period,
        last_timestamp_from_db
        )
    db.write_ohlcv(
        exchange.exchange_id,
        pair,
        period,
        tohlcv_list
        )


def update_all_exchanges_pairs(period):
        for exchange_i in exchanges.values():
            for pair in exchange_i.pairs:
                update(exchange_i, pair, period)


def initialize_exchanges():
    for exchange_i in exchanges.values():
        for pair in exchange_i.pairs:
            for period in exchange_i.periods.keys():
                update(exchange_i, pair, period)


def initialize_scheduler():
    schedule.every().minute.at(":15").do(update_all_exchanges_pairs, period='1m')
    schedule.every().hour.at(":01").do(update_all_exchanges_pairs, period='1h')
    schedule.every().day.at("00:01").do(update_all_exchanges_pairs, period='1d')


def main():

    initialize_exchanges()
    initialize_scheduler()
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
