import math
from numba import cuda, float32, int32, jit
import numpy as np
from .ta_cuda import *
from .ta_kernels import *


def moving_average(tacuda, json_data, res_index):
    """
    Calculate the moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        moving_average_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["moving_average" + "." + str(window_i)]
    return names


def exponential_moving_average(tacuda, json_data, res_index):
    """
    Calculate the exponential moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        exponential_moving_average_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["exponential_moving_average" + "." + str(window_i)]
    return names


def momentum(tacuda, json_data, res_index):
    """
    Calculate the momentum for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, step_i in enumerate(json_data["windows"]):
        momentum_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            step_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["momentum" + "." + str(step_i)]
    return names


def rate_of_change(tacuda, json_data, res_index):
    """
    Calculate the rate of change for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        rate_of_change_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["rate_of_change" + "." + str(window_i)]
    return names


def average_true_range(tacuda, json_data, res_index):
    """
    Calculate the average true range for the given data.
    https://en.wikipedia.org/wiki/Average_true_range

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        average_true_range_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["average_true_range" + "." + str(window_i)]
    return names


def stochastic_oscillator_k(tacuda, json_data, res_index):
    """
    Calculate stochastic oscillator %K for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        stochastic_oscillator_k_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["stochastic_oscillator_k" + "." + str(window_i)]
    return names


def stochastic_oscillator_d_ma(tacuda, json_data, res_index):
    """
    Calculate stochastic oscillator %D with moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        stochastic_oscillator_d_ma_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["stochastic_oscillator_d_ma" + "." + str(window_i)]
    return names


def stochastic_oscillator_d_ema(tacuda, json_data, res_index):
    """
    Calculate stochastic oscillator %D with exponential moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        stochastic_oscillator_d_ema_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["stochastic_oscillator_d_ema" + "." + str(window_i)]
    return names


def trix(tacuda, json_data, res_index):
    """
    Calculate TRIX for given data.
    https://en.wikipedia.org/wiki/Trix_(technical_analysis)

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        trix_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["trix" + "." + str(window_i)]
    return names


def mass_index(tacuda, json_data, res_index):
    """
    Calculate the Mass Index for given data.
    https://en.wikipedia.org/wiki/Mass_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    mass_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
        tacuda.ohlcv,
        json_data["param"],
        tacuda.temp_arr[0],
        tacuda.temp_arr[1],
        tacuda.result_gpu_mem,
        res_index
        )
    return ["mass_index" + "." + str(json_data["param"])]


def vortex_indicator_plus(tacuda, json_data, res_index):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        vortex_indicator_plus_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["vortex_indicator_plus" + "." + str(window_i)]
    return names


def vortex_indicator_minus(tacuda, json_data, res_index):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        vortex_indicator_minus_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["vortex_indicator_minus" + "." + str(window_i)]
    return names


def relative_strength_index(tacuda, json_data, res_index):
    """
    Calculate Relative Strength Index(RSI) for given data.
    https://en.wikipedia.org/wiki/Relative_strength_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        relative_strength_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["relative_strength_index" + "." + str(window_i)]
    return names


def true_strength_index(tacuda, json_data, res_index):
    """
    Calculate True Strength Index (TSI) for given data.
    https://en.wikipedia.org/wiki/True_strength_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i in range(len(json_data["windows"])):
        true_strength_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            json_data["windows"][i],
            json_data["windows_2"][i],
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.temp_arr[2],
            tacuda.temp_arr[3],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["true_strength_index" + "." + str(json_data["windows"][i]) + "." + str(json_data["windows_2"][i])]
    return names


def accumulation_distribution(tacuda, json_data, res_index):
    """
    Calculate Accumulation/Distribution for given data.
    https://en.wikipedia.org/wiki/Accumulation/distribution_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    accumulation_distribution_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
        tacuda.ohlcv,
        tacuda.temp_arr[0],
        tacuda.result_gpu_mem,
        res_index
        )
    return ["accumulation_distribution"]


def chaikin_oscillator(tacuda, json_data, res_index):
    """
    Calculate Chaikin Oscillator for given data.
    https://en.wikipedia.org/wiki/Chaikin_Analytics

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    chaikin_oscillator_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
        tacuda.ohlcv,
        tacuda.temp_arr[0],
        tacuda.temp_arr[1],
        tacuda.temp_arr[2],
        tacuda.result_gpu_mem,
        res_index
        )
    return ["chaikin_oscillator"]


def chaikin_money_flow(tacuda, json_data, res_index):
    """
    Calculate Money Flow for given data.
    https://en.wikipedia.org/wiki/Chaikin_Analytics

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    chaikin_money_flow_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
        tacuda.ohlcv,
        tacuda.temp_arr[0],
        tacuda.result_gpu_mem,
        res_index
        )
    return ["chaikin_money_flow"]


def money_flow_index(tacuda, json_data, res_index):
    """
    Calculate Money Flow Index and Ratio for given data.
    https://en.wikipedia.org/wiki/Money_flow_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        money_flow_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["money_flow_index" + "." + str(window_i)]
    return names

def on_balance_volume(tacuda, json_data, res_index):
    """
    Calculate On-Balance Volume for given data.
    https://en.wikipedia.org/wiki/On-balance_volume

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    on_balance_volume_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
        tacuda.ohlcv,
        tacuda.result_gpu_mem,
        res_index
        )
    return ["on_balance_volume"]


def force_index(tacuda, json_data, res_index):
    """
    Calculate Force Index for given data.
    https://en.wikipedia.org/wiki/Force_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        force_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["force_index" + "." + str(window_i)]
    return names


def ease_of_movement(tacuda, json_data, res_index):
    """
    Calculate Ease of Movement for given data.
    https://en.wikipedia.org/wiki/Ease_of_movement

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        ease_of_movement_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["ease_of_movement" + "." + str(window_i)]
    return names


def standard_deviation(tacuda, json_data, res_index):
    """
    Calculate Standard Deviation for given data.
    https://en.wikipedia.org/wiki/Standard_deviation

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        standard_deviation_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["standard_deviation" + "." + str(window_i)]
    return names
    

def commodity_channel_index(tacuda, json_data, res_index):
    """
    Calculate Commodity Channel Index for given data.
    https://en.wikipedia.org/wiki/Commodity_channel_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        commodity_channel_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["commodity_channel_index" + "." + str(window_i)]
    return names


def keltner_channel_m(tacuda, json_data, res_index):
    """
    Calculate Keltner Channel Middle for given data.
    https://www.investopedia.com/terms/k/keltnerchannel.asp

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        keltner_channel_m_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["keltner_channel_m" + "." + str(window_i)]
    return names


def keltner_channel_u(tacuda, json_data, res_index):
    """
    Calculate Keltner Channel Up for given data.
    https://www.investopedia.com/terms/k/keltnerchannel.asp

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        keltner_channel_u_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["keltner_channel_u" + "." + str(window_i)]
    return names


def keltner_channel_d(tacuda, json_data, res_index):
    """
    Calculate Keltner Channel Down for given data.
    https://www.investopedia.com/terms/k/keltnerchannel.asp

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        keltner_channel_d_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["keltner_channel_d" + "." + str(window_i)]
    return names

def ultimate_oscillator(tacuda, json_data, res_index):
    """
    Calculate Ultimate Oscillator for given data.
    https://en.wikipedia.org/wiki/Ultimate_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    ultimate_oscillator_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
        tacuda.ohlcv,
        tacuda.temp_arr[0],
        tacuda.temp_arr[1],
        tacuda.result_gpu_mem,
        res_index
        )
    return ["ultimate_oscillator"]


def donchian_channel_u(tacuda, json_data, res_index):
    """
    Calculate Donchian Channel Up of given data.
    https://www.investopedia.com/terms/d/donchianchannels.asp

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        donchian_channel_u_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["donchian_channel_u" + "." + str(window_i)]
    return names


def donchian_channel_d(tacuda, json_data, res_index):
    """
    Calculate Donchian Channel Down of given data.
    https://www.investopedia.com/terms/d/donchianchannels.asp

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        donchian_channel_d_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["donchian_channel_d" + "." + str(window_i)]
    return names


def donchian_channel_m(tacuda, json_data, res_index):
    """
    Calculate Donchian Channel Middle of given data.
    https://www.investopedia.com/terms/d/donchianchannels.asp

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        donchian_channel_m_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["donchian_channel_m" + "." + str(window_i)]
    return names


def bollinger_bands_m(tacuda, json_data, res_index):
    """
    Calculate Bollinger bands Middle for the given data.
    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        bollinger_bands_m_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["bollinger_bands_m" + "." + str(window_i)]
    return names


def bollinger_bands_u(tacuda, json_data, res_index):
    """
    Calculate Bollinger bands Up for the given data.
    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        bollinger_bands_u_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["bollinger_bands_u" + "." + str(window_i)]
    return names
        


def bollinger_bands_d(tacuda, json_data, res_index):
    """
    Calculate Bollinger bands Down for the given data.
    https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    names = []
    for i, window_i in enumerate(json_data["windows"]):
        bollinger_bands_d_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )
        names += ["bollinger_bands_d" + "." + str(window_i)]
    return names