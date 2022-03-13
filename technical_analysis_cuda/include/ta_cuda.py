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


##################################
# CUDA preparation functions
##################################

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
    Print cuda and GPU information
    """
    os.system('nvcc --version')
    os.system('nvidia-smi')
    print(cuda.detect())


##################################
# Moving Average
##################################


@cuda.jit('void(float64[:,:], int64, float64[:])')
def moving_average_kernel(ohlcv, window_size, out):
    """
    Calculate the moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    res = 0.0

    if (pos < ohlcv.shape[0]) and (n > 0):
        for i in range(1 - n, 1):
            res += ohlcv[pos + i, OHLCV_CLOSE]
    
        out[pos] = res / n


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def moving_average_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate the moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    res = 0.0
    
    if (pos < ohlcv.shape[0]) and (n > 0):
        for i in range(1 - n, 1):
            res += ohlcv[pos + i, OHLCV_CLOSE]
    
        out[pos, res_index] = res / n


def moving_average(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        moving_average_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)


##################################
# Exponential Moving Average
##################################


@cuda.jit('void(float64[:,:], int64, float64[:])')
def exponential_moving_average_kernel(ohlcv, window_size, out):
    """
    Calculate the exponential moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
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
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
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
    """
    Calculate the exponential moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        exponential_moving_average_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)


##################################
# Momentum
##################################


@cuda.jit('void(float64[:,:], int64, float64[:])')
def momentum_kernel(ohlcv, step, out):
    """
    Calculate the momentum for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)
    res = 0.0

    if step <= pos < ohlcv.shape[0]:
        res = ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - step, OHLCV_CLOSE]

        out[pos] = res


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def momentum_kernel_n(ohlcv, step, out, res_index):
    """
    Calculate the momentum for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    res = 0.0

    if step <= pos < ohlcv.shape[0]:
        res = ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - step, OHLCV_CLOSE]

        out[pos, res_index] = res


def momentum(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the momentum for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of steps between values
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        momentum_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)


##################################
# Rate of Change
##################################


@cuda.jit('void(float64[:,:], int64, float64[:])')
def rate_of_change_kernel(ohlcv, window_size, out):
    """
    Calculate the rate of change for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)
    res = 0.0

    if window_size <= pos < ohlcv.shape[0]:
        res = (ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - window_size + 1, OHLCV_CLOSE]) / ohlcv[pos - window_size + 1, OHLCV_CLOSE]

        out[pos] = res


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def rate_of_change_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate the rate of change for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    res = 0.0

    if window_size <= pos < ohlcv.shape[0]:
        res = (ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - window_size + 1, OHLCV_CLOSE]) / ohlcv[pos - window_size + 1, OHLCV_CLOSE]
    
        out[pos, res_index] = res


def rate_of_change(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the rate of change for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of steps between values
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        rate_of_change_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)


##################################
# Average True Range
##################################


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:])')
def average_true_range_kernel(ohlcv, window_size, temp_arr, out):
    """
    Calculate the average true range for the given data.
    https://en.wikipedia.org/wiki/Average_true_range

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)
    res = 0.0

    if pos < ohlcv.shape[0]:
        temp_arr[pos] = max(ohlcv[pos, OHLCV_HIGH], ohlcv[pos - 1, OHLCV_CLOSE]) - min(ohlcv[pos, OHLCV_LOW], ohlcv[pos - 1, OHLCV_CLOSE])

        cuda.syncthreads()

        res = temp_arr[pos - n]
        for i in range(1 - n, 1):
            res = temp_arr[pos + i] * k + res * (1.0 - k)
    
        out[pos] = res


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:,:], int64)')
def average_true_range_kernel_n(ohlcv, window_size, temp_arr, out, res_index):
    """
    Calculate the average true range for the given data.
    https://en.wikipedia.org/wiki/Average_true_range

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)
    res = 0.0

    if pos < ohlcv.shape[0]:
        temp_arr[pos] = max(ohlcv[pos, OHLCV_HIGH], ohlcv[pos - 1, OHLCV_CLOSE]) - min(ohlcv[pos, OHLCV_LOW], ohlcv[pos - 1, OHLCV_CLOSE])

        cuda.syncthreads()

        res = temp_arr[pos - n]
        for i in range(1 - n, 1):
            res = temp_arr[pos + i] * k + res * (1.0 - k)
    
        out[pos, res_index] = res


def average_true_range(ohlcv, windows, temp_arr, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the average true range for the given data.
    https://en.wikipedia.org/wiki/Average_true_range

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        average_true_range_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, temp_arr, out, res_index + i)


##################################
# Stochastic Oscillator
##################################


@cuda.jit('void(float64[:,:], int64, float64[:])')
def stochastic_oscillator_k_kernel(ohlcv, window_size, out):
    """
    Calculate stochastic oscillator %K for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)

    n = min(window_size, pos)
    res = 0.0

    if pos < ohlcv.shape[0]:
        low_n = ohlcv[pos, OHLCV_LOW]
        high_n = ohlcv[pos, OHLCV_HIGH]

        for i in range(1 - n, 0):
            low_n = min(low_n, ohlcv[pos + i, OHLCV_LOW])
            high_n = max(high_n, ohlcv[pos + i, OHLCV_HIGH])

        diff = high_n - low_n
        if diff != 0.0:
            res = (ohlcv[pos, OHLCV_CLOSE] - low_n) / diff * 100.0
        
        out[pos] = res


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def stochastic_oscillator_k_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate stochastic oscillator %K for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    res = 0.0
    if pos < ohlcv.shape[0]:
        low_n = ohlcv[pos, OHLCV_LOW]
        high_n = ohlcv[pos, OHLCV_HIGH]

        for i in range(1 - n, 0):
            low_n = min(low_n, ohlcv[pos + i, OHLCV_LOW])
            high_n = max(high_n, ohlcv[pos + i, OHLCV_HIGH])
        
        diff = high_n - low_n
        if diff != 0.0:
            res = (ohlcv[pos, OHLCV_CLOSE] - low_n) / diff * 100.0
        
        out[pos, res_index] = res


def stochastic_oscillator_k(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate stochastic oscillator %K for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        stochastic_oscillator_k_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:])')
def stochastic_oscillator_d_ma_kernel(ohlcv, window_size, temp_arr, out):
    """
    Calculate stochastic oscillator %D with moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)

    n = min(window_size, pos)
    temp_res = 0.0
    res = 0.0

    if (pos < ohlcv.shape[0]) and (n > 0):
        low_n = ohlcv[pos, OHLCV_LOW]
        high_n = ohlcv[pos, OHLCV_HIGH]

        for i in range(1 - n, 0):
            low_n = min(low_n, ohlcv[pos + i, OHLCV_LOW])
            high_n = max(high_n, ohlcv[pos + i, OHLCV_HIGH])

        diff = high_n - low_n
        if diff != 0.0:
            temp_res = (ohlcv[pos, OHLCV_CLOSE] - low_n) / diff * 100.0
        
        temp_arr[pos] = temp_res

        cuda.syncthreads()

        for i in range(1 - n, 1):
            res += temp_arr[pos + i]
    
        out[pos] = res / n


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:,:], int64)')
def stochastic_oscillator_d_ma_kernel_n(ohlcv, window_size, temp_arr, out, res_index):
    """
    Calculate stochastic oscillator %D with moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    temp_res = 0.0
    res = 0.0

    if (pos < ohlcv.shape[0]) and (n > 0):
        low_n = ohlcv[pos, OHLCV_LOW]
        high_n = ohlcv[pos, OHLCV_HIGH]

        for i in range(1 - n, 0):
            low_n = min(low_n, ohlcv[pos + i, OHLCV_LOW])
            high_n = max(high_n, ohlcv[pos + i, OHLCV_HIGH])

        diff = high_n - low_n
        if diff != 0.0:
            temp_res = (ohlcv[pos, OHLCV_CLOSE] - low_n) / diff * 100.0
        
        temp_arr[pos] = temp_res

        cuda.syncthreads()

        for i in range(1 - n, 1):
            res += temp_arr[pos + i]
    
        out[pos, res_index] = res / n


def stochastic_oscillator_d_ma(ohlcv, windows, temp_arr, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate stochastic oscillator %D with moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        stochastic_oscillator_d_ma_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, temp_arr, out, res_index + i)


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:])')
def stochastic_oscillator_d_ema_kernel(ohlcv, window_size, temp_arr, out):
    """
    Calculate stochastic oscillator %D with exponential moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)

    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)
    temp_res = 0.0
    res = 0.0

    if (pos < ohlcv.shape[0]) and (n > 0):
        low_n = ohlcv[pos, OHLCV_LOW]
        high_n = ohlcv[pos, OHLCV_HIGH]

        for i in range(1 - n, 0):
            low_n = min(low_n, ohlcv[pos + i, OHLCV_LOW])
            high_n = max(high_n, ohlcv[pos + i, OHLCV_HIGH])

        diff = high_n - low_n
        if diff != 0.0:
            temp_res = (ohlcv[pos, OHLCV_CLOSE] - low_n) / diff * 100.0
        
        temp_arr[pos] = temp_res

        cuda.syncthreads()

        res = temp_arr[pos - n]

        for i in range(1 - n, 1):
            res = temp_arr[pos + i] * k + res * (1.0 - k)
    
        out[pos] = res


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:,:], int64)')
def stochastic_oscillator_d_ema_kernel_n(ohlcv, window_size, temp_arr, out, res_index):
    """
    Calculate stochastic oscillator %D with exponential moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)
    temp_res = 0.0
    res = 0.0

    if (pos < ohlcv.shape[0]) and (n > 0):
        low_n = ohlcv[pos, OHLCV_LOW]
        high_n = ohlcv[pos, OHLCV_HIGH]

        for i in range(1 - n, 0):
            low_n = min(low_n, ohlcv[pos + i, OHLCV_LOW])
            high_n = max(high_n, ohlcv[pos + i, OHLCV_HIGH])

        diff = high_n - low_n
        if diff != 0.0:
            temp_res = (ohlcv[pos, OHLCV_CLOSE] - low_n) / diff * 100.0
        
        temp_arr[pos] = temp_res

        cuda.syncthreads()

        res = temp_arr[pos - n]

        for i in range(1 - n, 1):
            res = temp_arr[pos + i] * k + res * (1.0 - k)
    
        out[pos, res_index] = res


def stochastic_oscillator_d_ema(ohlcv, windows, temp_arr, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate stochastic oscillator %D with exponential moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param temp_arr: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        stochastic_oscillator_d_ema_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, temp_arr, out, res_index + i)