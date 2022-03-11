import math
import os
import json
from numba import cuda, float32, int32, jit
import pandas as pd
import numpy as np
import pymongo
from include.ta_cuda import *

np.set_printoptions(suppress=True)

mongo_username = os.environ.get("MONGO_USERNAME")
mongo_password = os.environ.get("MONGO_PASSWORD")

mongo_client = pymongo.MongoClient(
    "mongodb://" + str(mongo_username) + ":" + str(mongo_password) + "@mongodb:27017")
mongo_db = mongo_client["trading"]

"""def moving_average(a, n=24):
        ret = np.cumsum(a, dtype=np.float64)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n"""

with open('config.json') as json_file:
    json_data = json.load(json_file)
    exchange_name = json_data.get("exchange") # name from ccxt library
    symbol = json_data.get("symbol") # format - BTC/USDT
    period = json_data.get("period") # format - 1m, 1d,...

    print(json_data["technical indicators"][0]["function"])

    res = mongo_db["ohlcv"][exchange_name][symbol][period].find().sort([("timestamp", pymongo.ASCENDING)])
    res_df = pd.DataFrame(list(res))
    timeline = res_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_numpy()

    cuda_device_info()


    threads_per_block = 32
    blocks_per_grid = cuda_blocks_per_grid(len(res_df), threads_per_block)
  

    print(timeline)

    timeline_gpu_mem = cuda.to_device(timeline)
    result_gpu_mem = cuda.device_array(shape=(timeline.shape[0], 4), dtype=np.float64)
    
    res_index = 0
    
    for i, function_i in enumerate(json_data["technical indicators"]):
        ta_function = function_i["function"]
        ta_params = {
            "ohlcv": timeline_gpu_mem,
            "windows": function_i["windows"],
            "out": result_gpu_mem,
            "res_index": res_index,
            "blocks_per_grid": blocks_per_grid,
            "threads_per_block": threads_per_block
            }
        globals()[ta_function](**ta_params)
        res_index += len(function_i["windows"])
    
    

    res_row = np.array(result_gpu_mem.copy_to_host())
    close_row = res_df[['close']].to_numpy()

    print(close_row[-10:-1])
    
    print(res_row)