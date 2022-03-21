import os
import json
"""from numba import cuda, float32, int32, jit"""
import numpy as np
import pandas as pd
import pymongo
from tacuda import *

np.set_printoptions(suppress=True)

def main():

    mongo_username = os.environ.get("MONGO_USERNAME")
    mongo_password = os.environ.get("MONGO_PASSWORD")

    mongo_client = pymongo.MongoClient(
        "mongodb://" + str(mongo_username) + ":" + str(mongo_password) + "@mongodb:27017")
    mongo_db = mongo_client["trading"]

    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange") # name from ccxt library
        symbol = json_data.get("symbol") # format - BTC/USDT
        period = json_data.get("period") # format - 1m, 1d,...

        #print(json_data["technical indicators"][0]["function"])

        res = mongo_db[exchange_name][symbol][period]["ohlcv"].find().sort([("timestamp", pymongo.ASCENDING)])
        res_df = pd.DataFrame(list(res))
        timeline = res_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_numpy()

        print(timeline)

        ta_cuda = TACUDA(timeline, json_data)
        ta_cuda.cuda_device_info()
        
        names = ta_cuda.process()
        
        res_row = np.array(ta_cuda.result_gpu_mem.copy_to_host())
        
        ts = res_df[['timestamp']].to_numpy()
        
        res = np.concatenate((ts, res_row), axis=1)

        print(res)
        
        for i in range(1, res.shape[1]):
            tss = [{"timestamp": res[j, 0], "data": res[j, i]} for j in range(res.shape[0])]
            mongo_db[exchange_name][symbol][period][names[i - 1]].insert_many(tss)


if __name__ == '__main__':
    main()