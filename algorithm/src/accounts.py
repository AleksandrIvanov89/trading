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



logger = Logger("/logs/logs.log")

data_service_api = DataServiceAPI(
    os.environ.get("REST_API_BASE_URL"),
    os.environ.get("REST_API_USER"),
    os.environ.get("REST_API_PASSWORD"),
    logger
    )

db = MongoDB(os.environ.get("MONGO_USERNAME"), os.environ.get("MONGO_PASSWORD"))

accounts = {account['_id']: Account(account['_id'], logger) for account in db.get_all_accounts()}

def main():
    pass

if __name__ == '__main__':
    main()