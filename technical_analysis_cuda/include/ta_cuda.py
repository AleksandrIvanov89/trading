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


##################################
# TRIX 
##################################


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:], float64[:])')
def trix_kernel(ohlcv, window_size, temp_arr_1, temp_arr_2, out):
    """
    Calculate TRIX for given data.
    https://en.wikipedia.org/wiki/Trix_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr_1: temporary array
    :param temp_arr_2: temporary array
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)

    if pos < ohlcv.shape[0]:
        temp_res = ohlcv[pos - n, OHLCV_CLOSE]
        for i in range(1 - n, 1):
            temp_res = ohlcv[pos + i, OHLCV_CLOSE] * k + temp_res * (1.0 - k)
        temp_arr_1[pos] = temp_res

        cuda.syncthreads()

        temp_res = temp_arr_1[pos - n]
        for i in range(1 - n, 1):
            temp_res = temp_arr_1[pos + i] * k + temp_res * (1.0 - k)
        temp_arr_2[pos] = temp_res

        cuda.syncthreads()

        temp_res = temp_arr_2[pos - n]
        for i in range(1 - n, 1):
            temp_res = temp_arr_2[pos + i] * k + temp_res * (1.0 - k)
        temp_arr_1[pos] = temp_res

        cuda.syncthreads()

        temp_res = 0.0
        if temp_arr_1[pos - 1] != 0:
            temp_res = (temp_arr_1[pos] - temp_arr_1[pos - 1]) / temp_arr_1[pos - 1]
        
        out[pos] = temp_res


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:], float64[:,:], int64)')
def trix_kernel_n(ohlcv, window_size, temp_arr_1, temp_arr_2, out, res_index):
    """
    Calculate TRIX for given data.
    https://en.wikipedia.org/wiki/Trix_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr_1: temporary array
    :param temp_arr_2: temporary array
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)

    if pos < ohlcv.shape[0]:
        temp_res = ohlcv[pos - n, OHLCV_CLOSE]
        for i in range(1 - n, 1):
            temp_res = ohlcv[pos + i, OHLCV_CLOSE] * k + temp_res * (1.0 - k)
        temp_arr_1[pos] = temp_res

        cuda.syncthreads()

        temp_res = temp_arr_1[pos - n]
        for i in range(1 - n, 1):
            temp_res = temp_arr_1[pos + i] * k + temp_res * (1.0 - k)
        temp_arr_2[pos] = temp_res

        cuda.syncthreads()

        temp_res = temp_arr_2[pos - n]
        for i in range(1 - n, 1):
            temp_res = temp_arr_2[pos + i] * k + temp_res * (1.0 - k)
        temp_arr_1[pos] = temp_res

        cuda.syncthreads()

        temp_res = 0.0
        if temp_arr_1[pos - 1] != 0:
            temp_res = (temp_arr_1[pos] - temp_arr_1[pos - 1]) / temp_arr_1[pos - 1]

        out[pos, res_index] = temp_res


def trix(ohlcv, windows, temp_arr_1, temp_arr_2, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate TRIX for given data.
    https://en.wikipedia.org/wiki/Trix_(technical_analysis)

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param temp_arr_1: link to GPU memory with temporary array for calculations
    :param temp_arr_2: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        trix_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, temp_arr_1, temp_arr_2, out, res_index + i)


##################################
# Mass Index 
##################################


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:], float64[:])')
def mass_index_kernel(ohlcv, periods_per_day, temp_arr_1, temp_arr_2, out):
    """
    Calculate the Mass Index for given data.
    https://en.wikipedia.org/wiki/Mass_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param periods_per_day: number of periods in 1 day
    :param temp_arr_1: link to GPU memory with temporary array for calculations
    :param temp_arr_2: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    """
    pos = cuda.grid(1)
    n = min(9, pos)
    k = 2.0 / (n + 1.0)

    if pos < ohlcv.shape[0]:
        temp_arr_1[pos] = ohlcv[pos - n, OHLCV_HIGH] - ohlcv[pos - n, OHLCV_LOW]

        for i in range(1 - n, 1):
            temp_arr_1[pos] = (ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i, OHLCV_LOW]) * k + temp_arr_1[pos] * (1.0 - k)

    cuda.syncthreads()

    if pos < ohlcv.shape[0]:
        temp_arr_2[pos] = temp_arr_1[pos - n]
        for i in range(1 - n, 1):
            temp_arr_2[pos] = temp_arr_1[pos + i] * k + temp_arr_2[pos] * (1.0 - k)

    cuda.syncthreads()

    if 24 < pos < ohlcv.shape[0]:
        out[pos] = 0.0
        for i in range(-24, 1):
            out[pos] += temp_arr_1[pos + i] / temp_arr_2[pos + i]
    
        out[pos] /= periods_per_day


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:], float64[:,:], int64)')
def mass_index_kernel_n(ohlcv, periods_per_day, temp_arr_1, temp_arr_2, out, res_index):
    """
    Calculate the Mass Index for given data.
    https://en.wikipedia.org/wiki/Mass_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param periods_per_day: number of periods in 1 day
    :param temp_arr_1: link to GPU memory with temporary array for calculations
    :param temp_arr_2: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(9 * periods_per_day, pos)
    k = 2.0 / (n + 1.0)

    if pos < ohlcv.shape[0]:
        temp_arr_1[pos] = ohlcv[pos - n, OHLCV_HIGH] - ohlcv[pos - n, OHLCV_LOW]

        for i in range(1 - n, 1):
            temp_arr_1[pos] = (ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i, OHLCV_LOW]) * k + temp_arr_1[pos] * (1.0 - k)

    cuda.syncthreads()

    if pos < ohlcv.shape[0]:
        temp_arr_2[pos] = temp_arr_1[pos - n]
        for i in range(1 - n, 1):
            temp_arr_2[pos] = temp_arr_1[pos + i] * k + temp_arr_2[pos] * (1.0 - k)

    cuda.syncthreads()

    if 24 * periods_per_day < pos < ohlcv.shape[0]:
        out[pos, res_index] = 0.0
        for i in range(-24 * periods_per_day, 1):
            out[pos, res_index] += temp_arr_1[pos + i] / temp_arr_2[pos + i]
        out[pos, res_index] /= periods_per_day


def mass_index(ohlcv, param, temp_arr_1, temp_arr_2, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the Mass Index for given data.
    https://en.wikipedia.org/wiki/Mass_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param param: number of periods in 1 day
    :param temp_arr_1: link to GPU memory with temporary array for calculations
    :param temp_arr_2: link to GPU memory with temporary array for calculations
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    mass_index_kernel_n[blocks_per_grid, threads_per_block](ohlcv, param, temp_arr_1, temp_arr_2, out, res_index)


##################################
# Vortex Indicator
##################################


@cuda.jit('void(float64[:,:], int64, float64[:])')
def vortex_indicator_plus_kernel(ohlcv, window_size, out):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    
    if n + 1 < pos < ohlcv.shape[0]:
        true_range = 0.0
        vortex_movement_plus = 0.0
        for i in range(-n, 1):
            true_range += max(
                ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i, OHLCV_LOW],
                abs(ohlcv[pos + i, OHLCV_LOW] - ohlcv[pos + i - 1, OHLCV_CLOSE]),
                abs(ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i - 1, OHLCV_CLOSE]))
            vortex_movement_plus += abs(ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i - 1, OHLCV_LOW])
        if true_range != 0.0:
            out[pos] = vortex_movement_plus / true_range


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def vortex_indicator_plus_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    
    if n + 1 < pos < ohlcv.shape[0]:
        true_range = 0.0
        vortex_movement_plus = 0.0
        for i in range(-n, 1):
            true_range += max(
                ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i, OHLCV_LOW],
                abs(ohlcv[pos + i, OHLCV_LOW] - ohlcv[pos + i - 1, OHLCV_CLOSE]),
                abs(ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i - 1, OHLCV_CLOSE]))
            vortex_movement_plus += abs(ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i - 1, OHLCV_LOW])
        if true_range != 0.0:
            out[pos, res_index] = vortex_movement_plus / true_range


def vortex_indicator_plus(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        vortex_indicator_plus_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)


@cuda.jit('void(float64[:,:], int64, float64[:])')
def vortex_indicator_minus_kernel(ohlcv, window_size, out):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: result
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    
    if n + 1 < pos < ohlcv.shape[0]:
        true_range = 0.0
        vortex_movement_minus = 0.0
        for i in range(-n, 1):
            true_range += max(
                ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i, OHLCV_LOW],
                abs(ohlcv[pos + i, OHLCV_LOW] - ohlcv[pos + i - 1, OHLCV_CLOSE]),
                abs(ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i - 1, OHLCV_CLOSE]))
            vortex_movement_minus += abs(ohlcv[pos + i, OHLCV_LOW] - ohlcv[pos + i - 1, OHLCV_HIGH])
        if true_range != 0.0:
            out[pos] = vortex_movement_minus / true_range


@cuda.jit('void(float64[:,:], int64, float64[:,:], int64)')
def vortex_indicator_minus_kernel_n(ohlcv, window_size, out, res_index):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n = min(window_size, pos)
    
    if n + 1 < pos < ohlcv.shape[0]:
        true_range = 0.0
        vortex_movement_minus = 0.0
        for i in range(-n, 1):
            true_range += max(
                ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i, OHLCV_LOW],
                abs(ohlcv[pos + i, OHLCV_LOW] - ohlcv[pos + i - 1, OHLCV_CLOSE]),
                abs(ohlcv[pos + i, OHLCV_HIGH] - ohlcv[pos + i - 1, OHLCV_CLOSE]))
            vortex_movement_minus += abs(ohlcv[pos + i, OHLCV_LOW] - ohlcv[pos + i - 1, OHLCV_HIGH])
        if true_range != 0.0:
            out[pos, res_index] = vortex_movement_minus / true_range


def vortex_indicator_minus(ohlcv, windows, out, res_index, blocks_per_grid, threads_per_block):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param out: link to GPU memory for result
    :param res_index: column index in result array
    :param blocks_per_grid: CUDA blocks per grid
    :param threads_per_block: CUDA threads per block
    """
    for i, window_i in enumerate(windows):
        vortex_indicator_minus_kernel_n[blocks_per_grid, threads_per_block](ohlcv, window_i, out, res_index + i)