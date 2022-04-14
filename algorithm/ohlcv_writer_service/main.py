import os
import json
import time

from exchanges_db import *


def load_config():
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange")  # name from ccxt library
        symbol = json_data.get("symbol")  # format - BTC/USDT
        return exchange_name, symbol


def main():
    exchange_name, symbol = load_config()
    #os.environ.get("FIREBASE_CREDENTIALS_PATH"), os.environ.get("FIREBASE_BOT_ID")
    mongo_username = os.environ.get("MONGO_USERNAME")
    mongo_password = os.environ.get("MONGO_PASSWORD")
    rest_api_user = os.environ.get("REST_API_USER")
    rest_api_password = os.environ.get("REST_API_PASSWORD")
    rest_api_base_url = os.environ.get("REST_API_BASE_URL")
    data_service_api = DataServiceAPI(
        rest_api_base_url, rest_api_user, rest_api_password)
    mongodb = MongoDB(exchange_name, symbol, mongo_username,
                      mongo_password, "mongodb:27017", data_service_api)

    mongodb.update_ohlcv_loop()


if __name__ == '__main__':
    main()
