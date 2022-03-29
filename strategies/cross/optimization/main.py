import os
import json
import numpy as np
import pandas as pd
import pymongo

import math
from numba import cuda, float32, int32, jit


OHLCV_TIMESTAMP = 0
OHLCV_OPEN = 1
OHLCV_HIGH = 2
OHLCV_LOW = 3
OHLCV_CLOSE = 4
OHLCV_VOLUME = 5

MAX_WINDOW_SIZE = 200 * 24 * 60
STEP = 24 * 60

mongo_username = os.environ.get("MONGO_USERNAME")
mongo_password = os.environ.get("MONGO_PASSWORD")

mongo_client = pymongo.MongoClient(
    "mongodb://" + str(mongo_username) + ":" + str(mongo_password) + "@mongodb:27017")
mongo_db = mongo_client["trading"]


def cuda_blocks_per_grid(length, threads_per_block):
    return math.ceil(length / threads_per_block)


def cuda_blocks_per_grid_2d(shape, threads_per_block_2d):
    return (int(math.ceil(shape[0] / threads_per_block_2d[0])),
            int(math.ceil(shape[1] / threads_per_block_2d[1])))


@cuda.jit('void(float64[:,:], int64[:], float64[:,:])')
def moving_average_kernel_2d(ohlcv, window_size, out):
    """
    Calculate the moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    """
    thread_x, thread_y = cuda.grid(2)

    n = min(window_size[thread_y], thread_x)
    res = 0.0
    
    if (thread_x < ohlcv.shape[0]) and (thread_y < window_size.shape[0]) and (n > 0):
        for i in range(1 - n, 1):
            res += ohlcv[thread_x + i, OHLCV_CLOSE]
        out[thread_x, thread_y] = res / n


@cuda.jit('void(float64[:,:], float64[:,:], int64[:,:], float64[:,:])')
def cross_sim(ohlcv, ma, algo_params, result):
    # params initialization
    pos = cuda.grid(1)
    balance_btc = 1.0
    balance_usdt = 0.0
    base_t = MAX_WINDOW_SIZE
    mod_n = 3
    prev_t = base_t
    
    if pos < algo_params.shape[0]:
        result[pos, 0] = balance_btc * ohlcv[base_t, OHLCV_CLOSE] + balance_usdt
        res_t = 1

        window_short = algo_params[pos, 0]
        window_long = algo_params[pos, 1]

        prev_cross_temp = 0
        for i in range(0, mod_n):
            temp_1 = ma[base_t - i, window_short] - ma[base_t - i, window_long]
            if temp_1 > 0:
                prev_cross_temp += 1
            elif temp_1 < 0:
                prev_cross_temp -= 1

        for t in range(base_t + mod_n + 1, ohlcv.shape[0] - 1):
            #algorithm logic
            cross_temp = 0
            for i in range(0, mod_n):
                #filter
                temp_1 = ma[t - i, window_short] - ma[t - i, window_long]
                if temp_1 > 0:
                    cross_temp += 1
                elif temp_1 < 0:
                    cross_temp -= 1
            
            if (prev_cross_temp > 0) and (cross_temp < 0) and (balance_btc > 0.0):
                # sell
                balance_usdt = balance_btc * ohlcv[t + 1, OHLCV_OPEN] * (1.0 - 0.002)
                balance_btc = 0.0
                
            if (prev_cross_temp < 0) and (cross_temp > 0) and (balance_usdt > 0.0):
                # buy
                balance_btc = balance_usdt / ohlcv[t + 1, OHLCV_OPEN] * (1.0 - 0.002)
                balance_usdt = 0.0

            prev_cross_temp = cross_temp

            # save results
            if t - prev_t == STEP:
                result[pos, res_t] = balance_btc * ohlcv[t, OHLCV_CLOSE] + balance_usdt
                res_t += 1
                prev_t = t


def load_timeseries():
    """
    Load timeseries from mongodb
    """
    
    with open('config.json') as json_file:
        json_data = json.load(json_file)
        exchange_name = json_data.get("exchange") # name from ccxt library
        symbol = json_data.get("symbol") # format - BTC/USDT
        period = json_data.get("period") # format - 1m, 1d,...

        res = mongo_db[exchange_name][symbol][period]["ohlcv"].find()
        res_df = pd.DataFrame(list(res))
        timeseries = res_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_numpy()

        return timeseries


def main():
    
    window_short = np.array([5, 10, 15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 300, 330, 360, 390, 420, 480, 540, 600, 720, 980], dtype=np.int64)
    windows_long = np.linspace(STEP, MAX_WINDOW_SIZE, 200, endpoint=True, dtype=np.int64)
    windows = np.concatenate((window_short, windows_long), axis=0)
    params = []
    for i in range(windows.shape[0]):
        for j in range(i + 1, windows.shape[0]):
            params += [[i, j]]
    np_params = np.array(params, dtype=np.int64)
    print(windows)
    print(np_params.shape[0])

    ts = load_timeseries()
    timeseries = ts[ts[:, 0].argsort()]
    print(timeseries)
    print(timeseries[MAX_WINDOW_SIZE,:])

    # copy data to GPU memmory
    ohlcv_gpu = cuda.to_device(timeseries)
    windows_gpu = cuda.to_device(windows)
    params_gpu = cuda.to_device(params)
    # define space in GPU memmory for calculations and results
    ma_gpu = cuda.device_array(shape=(timeseries.shape[0], windows.shape[0]), dtype=np.float64)
    #result_gpu = cuda.device_array(shape=np_params.shape[0], dtype=np.float64)
    result_gpu = cuda.device_array(shape=(np_params.shape[0], (timeseries.shape[0] - MAX_WINDOW_SIZE) // STEP + 1), dtype=np.float64)
    
    threads_per_block_1 = (8, 8)
    blocks_per_grid_1 = cuda_blocks_per_grid_2d((timeseries.shape[0], windows.shape[0]), threads_per_block_1)

    threads_per_block_2 = 64
    blocks_per_grid_2 = cuda_blocks_per_grid(np_params.shape[0], threads_per_block_2)

    # calc all moving averages
    moving_average_kernel_2d[blocks_per_grid_1, threads_per_block_1](ohlcv_gpu, windows_gpu, ma_gpu)
    
    # simulate algorithm
    cross_sim[blocks_per_grid_2, threads_per_block_2](ohlcv_gpu, ma_gpu, params_gpu, result_gpu)
    
    # get results from GPU memmory
    res_row = np.array(result_gpu.copy_to_host())
    print(res_row)
    print(res_row.shape)
    
    windows_out = np.array([[windows[i[0]], windows[i[1]]] for i in np_params], dtype=np.float64)

    # write results to mongodb
    db_res = [{
        "window_1": windows_out[i, 0],
        "window_2": windows_out[i, 1],
        "result": list(res_row[i,:])
        } for i in range(res_row.shape[0])]
    mongo_db["cross_ma_opt"].insert_many(db_res)

if __name__ == '__main__':
    main()