import math
from numba import cuda, float32, int32, jit


OHLCV_TIMESTAMP = 0
OHLCV_OPEN = 1
OHLCV_HIGH = 2
OHLCV_LOW = 3
OHLCV_CLOSE = 4
OHLCV_VOLUME = 5

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


##################################
# Relative Strength Index
##################################


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:], float64[:])')
def relative_strength_index_kernel(ohlcv, window_size, temp_arr_1, temp_arr_2, out):
    """
    Calculate Relative Strength Index(RSI) for given data.
    https://en.wikipedia.org/wiki/Relative_strength_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param temp_arr_1: link to GPU memory with temporary array for calculations
    :param temp_arr_2: link to GPU memory with temporary array for calculations
    :param out: result
    """
    pos = cuda.grid(1)
    temp_arr_1[pos] = 0.0
    temp_arr_2[pos] = 0.0
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)

    if 0 < pos < ohlcv.shape[0]:
        u = 0.0
        d = 0.0

        delta = ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - 1, OHLCV_CLOSE]
        if delta > 0.0:
            u = delta
        else:
            d = abs(delta)
        
        temp_arr_1[pos] = u
        temp_arr_2[pos] = d

        cuda.syncthreads()

        res_u = temp_arr_1[pos - n]
        res_d = temp_arr_2[pos - n]

        for i in range(1 - n, 1):
            res_u = temp_arr_1[pos + i] * k + res_u * (1.0 - k)
            res_d = temp_arr_2[pos + i] * k + res_d * (1.0 - k)
        
        if (res_d != 0.0) and (res_u / res_d != -1.0):
            out[pos] = 100.0 - 100.0 / (1.0 + res_u / res_d)


@cuda.jit('void(float64[:,:], int64, float64[:], float64[:], float64[:,:], int64)')
def relative_strength_index_kernel_n(ohlcv, window_size, temp_arr_1, temp_arr_2, out, res_index):
    """
    Calculate Relative Strength Index(RSI) for given data.
    https://en.wikipedia.org/wiki/Relative_strength_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param window: array of window sizes
    :param temp_arr_1: link to GPU memory with temporary array for calculations
    :param temp_arr_2: link to GPU memory with temporary array for calculations
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    temp_arr_1[pos] = 0.0
    temp_arr_2[pos] = 0.0
    n = min(window_size, pos)
    k = 2.0 / (n + 1.0)

    if 0 < pos < ohlcv.shape[0]:
        u = 0.0
        d = 0.0

        delta = ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - 1, OHLCV_CLOSE]
        if delta > 0.0:
            u = delta
        else:
            d = abs(delta)
        
        temp_arr_1[pos] = u
        temp_arr_2[pos] = d

        cuda.syncthreads()

        res_u = temp_arr_1[pos - n]
        res_d = temp_arr_2[pos - n]

        for i in range(1 - n, 1):
            res_u = temp_arr_1[pos + i] * k + res_u * (1.0 - k)
            res_d = temp_arr_2[pos + i] * k + res_d * (1.0 - k)
        
        if (res_d != 0.0) and (res_u / res_d != -1.0):
            out[pos, res_index] = 100.0 - 100.0 / (1.0 + res_u / res_d)


##################################
# True Strength Index
##################################


@cuda.jit('void(float64[:,:], int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:])')
def true_strength_index_kernel(ohlcv, r, s, temp_arr_1, temp_arr_2, temp_arr_3, temp_arr_4, out):
    """
    Calculate True Strength Index (TSI) for given data.
    https://en.wikipedia.org/wiki/True_strength_index

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr_1: temporary array
    :param temp_arr_2: temporary array
    :param temp_arr_3: temporary array
    :param temp_arr_4: temporary array
    :param out: result
    """
    pos = cuda.grid(1)
    n1 = min(r, pos)
    n2 = min(s, pos)
    k1 = 2.0 / (n1 + 1.0)
    k2 = 2.0 / (n2 + 1.0)

    if 0 < pos < ohlcv.shape[0]:
        delta = ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - 1, OHLCV_CLOSE]
        temp_arr_1[pos] = delta
        temp_arr_2[pos] = abs(delta)

        cuda.syncthreads()

        temp_arr_3[pos] = temp_arr_1[pos - n1]
        temp_arr_4[pos] = temp_arr_2[pos - n1]

        for i in range(1 - n1, 1):
            temp_arr_3[pos] = temp_arr_1[pos + i] * k1 + temp_arr_3[pos] * (1.0 - k1)
            temp_arr_4[pos] = temp_arr_2[pos + i] * k1 + temp_arr_4[pos] * (1.0 - k1)

        cuda.syncthreads()

        temp_arr_1[pos] = temp_arr_3[pos - n2]
        temp_arr_2[pos] = temp_arr_4[pos - n2]

        for i in range(1 - n2, 1):
            temp_arr_1[pos] = temp_arr_3[pos + i] * k2 + temp_arr_1[pos] * (1.0 - k1)
            temp_arr_2[pos] = temp_arr_4[pos + i] * k2 + temp_arr_2[pos] * (1.0 - k1)

        out[pos] = 100.0 * temp_arr_3[pos] / temp_arr_4[pos]


@cuda.jit('void(float64[:,:], int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:,:], int64)')
def true_strength_index_kernel_n(ohlcv, r, s, temp_arr_1, temp_arr_2, temp_arr_3, temp_arr_4, out, res_index):
    """
    Calculate True Strength Index (TSI) for given data.
    https://en.wikipedia.org/wiki/True_strength_index

    :param ohlcv: Open, High, Low, Close, Volume
    :param window_size: window size
    :param temp_arr_1: temporary array
    :param temp_arr_2: temporary array
    :param temp_arr_3: temporary array
    :param temp_arr_4: temporary array
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n1 = min(r, pos)
    n2 = min(s, pos)
    k1 = 2.0 / (n1 + 1.0)
    k2 = 2.0 / (n2 + 1.0)

    if 0 < pos < ohlcv.shape[0]:
        delta = ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos - 1, OHLCV_CLOSE]
        temp_arr_1[pos] = delta
        temp_arr_2[pos] = abs(delta)

        cuda.syncthreads()

        temp_arr_3[pos] = temp_arr_1[pos - n1]
        temp_arr_4[pos] = temp_arr_2[pos - n1]

        for i in range(1 - n1, 1):
            temp_arr_3[pos] = temp_arr_1[pos + i] * k1 + temp_arr_3[pos] * (1.0 - k1)
            temp_arr_4[pos] = temp_arr_2[pos + i] * k1 + temp_arr_4[pos] * (1.0 - k1)

        cuda.syncthreads()

        temp_arr_1[pos] = temp_arr_3[pos - n2]
        temp_arr_2[pos] = temp_arr_4[pos - n2]

        for i in range(1 - n2, 1):
            temp_arr_1[pos] = temp_arr_3[pos + i] * k2 + temp_arr_1[pos] * (1.0 - k1)
            temp_arr_2[pos] = temp_arr_4[pos + i] * k2 + temp_arr_2[pos] * (1.0 - k1)

        out[pos, res_index] = 100.0 * temp_arr_3[pos] / temp_arr_4[pos]


##################################
# Accumulation Distribution Index
##################################


@cuda.jit('void(float64[:,:], float64[:], float64[:])')
def accumulation_distribution_kernel(ohlcv, temp_arr, out):
    """
    Calculate Accumulation/Distribution for given data.
    https://en.wikipedia.org/wiki/Accumulation/distribution_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param temp_arr: temporary array
    :param out: result
    """
    pos = cuda.grid(1)
    
    if pos < ohlcv.shape[0]:
        if pos == 0:
            temp_arr[pos] = 0.0
        else:
            temp_arr[pos] = (2.0 * ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) / (ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) * ohlcv[pos, OHLCV_VOLUME]

        cuda.syncthreads()

        res = 0.0
        for i in range(pos + 1):
            res += temp_arr[i]

        out[pos] = res


@cuda.jit('void(float64[:,:], float64[:], float64[:,:], int64)')
def accumulation_distribution_kernel_n(ohlcv, temp_arr, out, res_index):
    """
    Calculate Accumulation/Distribution for given data.
    https://en.wikipedia.org/wiki/Accumulation/distribution_index

    :param ohlcv: link to GPU memory with array of timestamp, Open, High, Low, Close, Volume
    :param temp_arr: temporary array
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    
    if pos < ohlcv.shape[0]:
        if pos == 0:
            temp_arr[pos] = 0.0
        else:
            temp_arr[pos] = (2.0 * ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) / (ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) * ohlcv[pos, OHLCV_VOLUME]

        cuda.syncthreads()

        res = 0.0
        for i in range(pos + 1):
            res += temp_arr[i]
        
        out[pos, res_index] = res


##################################
# Chaikin Oscillator
##################################


@cuda.jit('void(float64[:,:], float64[:], float64[:], float64[:], float64[:])')
def chaikin_oscillator_kernel(ohlcv, temp_arr_1, temp_arr_2, temp_arr_3, out):
    """
    Calculate Chaikin Oscillator for given data.
    https://en.wikipedia.org/wiki/Chaikin_Analytics

    :param ohlcv: Open, High, Low, Close, Volume
    :param temp_arr_1: temporary array
    :param temp_arr_2: temporary array
    :param temp_arr_3: temporary array
    :param out: result
    """
    pos = cuda.grid(1)
    n1 = min(3, pos)
    n2 = min(10, pos)
    k1 = 2.0 / (n1 + 1.0)
    k2 = 2.0 / (n2 + 1.0)

    if pos < ohlcv.shape[0]:
        temp_arr_1[pos] = (2.0 * ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) / (ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) * ohlcv[pos, OHLCV_VOLUME]

        cuda.syncthreads()

        temp_arr_2[pos] = temp_arr_1[pos - n1]
        for i in range(1 - n1, 1):
            temp_arr_2[pos] = temp_arr_1[pos + i] * k1 + temp_arr_2[pos] * (1.0 - k1)

        temp_arr_3[pos] = temp_arr_1[pos - n2]
        for i in range(1 - n2, 1):
            temp_arr_3[pos] = temp_arr_1[pos + i] * k2 + temp_arr_3[pos] * (1.0 - k2)

        out[pos] = temp_arr_2[pos] - temp_arr_3[pos]


@cuda.jit('void(float64[:,:], float64[:], float64[:], float64[:], float64[:,:], int64)')
def chaikin_oscillator_kernel_n(ohlcv, temp_arr_1, temp_arr_2, temp_arr_3, out, res_index):
    """
    Calculate Chaikin Oscillator for given data.
    https://en.wikipedia.org/wiki/Chaikin_Analytics

    :param ohlcv: Open, High, Low, Close, Volume
    :param temp_arr_1: temporary array
    :param temp_arr_2: temporary array
    :param temp_arr_3: temporary array
    :param out: result array
    :param res_index: column index in result array
    """
    pos = cuda.grid(1)
    n1 = min(3, pos)
    n2 = min(10, pos)
    k1 = 2.0 / (n1 + 1.0)
    k2 = 2.0 / (n2 + 1.0)

    if pos < ohlcv.shape[0]:
        temp_arr_1[pos] = (2.0 * ohlcv[pos, OHLCV_CLOSE] - ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) / (ohlcv[pos, OHLCV_HIGH] - ohlcv[pos, OHLCV_LOW]) * ohlcv[pos, OHLCV_VOLUME]

        cuda.syncthreads()

        temp_arr_2[pos] = temp_arr_1[pos - n1]
        for i in range(1 - n1, 1):
            temp_arr_2[pos] = temp_arr_1[pos + i] * k1 + temp_arr_2[pos] * (1.0 - k1)

        temp_arr_3[pos] = temp_arr_1[pos - n2]
        for i in range(1 - n2, 1):
            temp_arr_3[pos] = temp_arr_1[pos + i] * k2 + temp_arr_3[pos] * (1.0 - k2)

        out[pos, res_index] = temp_arr_2[pos] - temp_arr_3[pos]
