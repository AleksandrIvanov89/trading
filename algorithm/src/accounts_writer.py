import os
import json
import time
import schedule

from libs import *

logger = Logger("/logs/logs.log")
    
data_service_api = DataServiceAPI(
    os.environ.get("ACCOUNTS_REST_API_BASE_URL"),
    os.environ.get("REST_API_USER"),
    os.environ.get("REST_API_PASSWORD"),
    logger
    )

db = MongoDB(
    os.environ.get("MONGO_USERNAME"),
    os.environ.get("MONGO_PASSWORD"),
    "mongodb:27017"
    )


def update_all_accounts():
    data = data_service_api.get_all_accounts_balances()
    db.write_accounts_balances(data)


def main():
    
    schedule.every().minute.at(":00").do(update_all_accounts)
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()