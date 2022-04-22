import os
import json
import time
import schedule

from exchanges_db import *

def load_config():
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange")  # name from ccxt library
        symbol = json_data.get("symbol")  # format - BTC/USDT
        return exchange_name, symbol


def main():
    exchange_name, symbol = load_config()
    logger = Logger("/logs/logs.log")
    
    data_service_api = DataServiceAPI(
        os.environ.get("REST_API_BASE_URL"),
        os.environ.get("REST_API_USER"),
        os.environ.get("REST_API_PASSWORD"),
        logger
        )

    db_list = [
        MongoDB(
            exchange_name,
            symbol,
            os.environ.get("MONGO_USERNAME"),
            os.environ.get("MONGO_PASSWORD"),
            "mongodb:27017",
            data_service_api,
            logger
        ),
        Firebase(
            exchange_name,
            symbol,
            os.environ.get("FIREBASE_CREDENTIALS_PATH"),
            data_service_api,
            logger
        )]
    
    def update(period):
        for db in db_list:
            db.update_ohlcv(period)
    
    schedule.every().minute.at(":15").do(update, period='1m')
    schedule.every().hour.at(":01").do(update, period='1h')
    schedule.every().day.at("00:01").do(update, period='1d')
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()
