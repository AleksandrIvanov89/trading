import math
from numba import cuda, float32, int32, jit
import os
import numpy as np

OHLCV_TIMESTAMP = 0
OHLCV_OPEN = 1
OHLCV_HIGH = 2
OHLCV_LOW = 3
OHLCV_CLOSE = 4
OHLCV_VOLUME = 5

def cuda_blocks_per_grid(length, threads_per_block):
    """
    Calculate the number of cuda blocks per grid for length of an array
    
    :param length: lenght of the array
    :param threads_per_block: the number of threads per block
    """
    return math.ceil(length / threads_per_block)


def cuda_blocks_per_grid_2d(shape, threads_per_block_2d):
    """
    Calculate the number of cuda blocks per grid for length of an 2d array

    :param shape: shape of the 2d array
    :param threads_per_block: the number of threads per block
    """
    return (int(math.ceil(shape[0] / threads_per_block_2d[0])),
            int(math.ceil(shape[1] / threads_per_block_2d[1])))


def cuda_device_info():
    """
    Print cuda and gpu information
    """
    os.system('nvcc --version')
    os.system('nvidia-smi')
    print(cuda.detect())

@cuda.jit('void(float64[:,:], int64, float64[:])')
def moving_average_kernel(ohlcv, window_size, out):
    """
    Calculate the moving average for the given data.

    :param ohlcv: link to gpu memory Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to gpu memory for result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    res = 0.0

    if (pos < ohlcv.shape[0]) and (n > 0):
        for i in range(1 - n, 1):
            res += ohlcv[pos + i, OHLCV_CLOSE]
        res /= n
    
    out[pos] = res

@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def moving_average_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate the moving average for the given data.

    :param ohlcv: link to gpu memory Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to gpu memory for result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    res = 0.0
    
    if (pos < ohlcv.shape[0]) and (n > 0):
        for i in range(1 - n, 1):
            res += ohlcv[pos + i, OHLCV_CLOSE]
        res /= n
    
    out[pos, res_index] = res

def moving_average(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    for i, window_i in enumerate(windows):
        moving_average_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)

@cuda.jit('void(float64[:,:], int64, float64[:])')
def exponential_moving_average_kernel(ohlcv, window_size, out):
    """
    Calculate the exponential moving average for the given data.

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)
    res = 0.0

    if pos < ohlcv.shape[0]:
        res = ohlcv[pos - n, OHLCV_CLOSE]

        for i in range(1 - n, 1):
            res = ohlcv[pos + i, OHLCV_CLOSE] * k + res * (1.0 - k)
    
    out[pos] = res


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def exponential_moving_average_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate the exponential moving average for the given data.

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)
    res = 0.0

    if pos < ohlcv.shape[0]:
        res = ohlcv[pos - n, OHLCV_CLOSE]

        for i in range(1 - n, 1):
            res = ohlcv[pos + i, OHLCV_CLOSE] * k + res * (1.0 - k)
    
    out[pos, res_index] = res

def exponential_moving_average(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    for i, window_i in enumerate(windows):
        exponential_moving_average_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)