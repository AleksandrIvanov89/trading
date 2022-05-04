import os
from abc import abstractmethod
from libs import *

logger = Logger("/logs/logs.log")
bot_id = os.environ.get("BOT_ID")

db = MongoDB(
    os.environ.get("MONGO_USERNAME"),
    os.environ.get("MONGO_PASSWORD"),
    "mongodb:27017",
    logger
    )

#getattr
algo = Cross_MA(db, bot_id, logger)


def main():
    algo.initialize()
    while True:
        algo.process()

if __name__ == "__main__":
    main()