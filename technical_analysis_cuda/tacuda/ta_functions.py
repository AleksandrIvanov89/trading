import math
from numba import cuda, float32, int32, jit
import os
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
    for i, window_i in enumerate(json_data["windows"]):
        moving_average_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def exponential_moving_average(tacuda, json_data, res_index):
    """
    Calculate the exponential moving average for the given data.
    https://en.wikipedia.org/wiki/Moving_average

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        exponential_moving_average_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def momentum(tacuda, json_data, res_index):
    """
    Calculate the momentum for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, step_i in enumerate(json_data["windows"]):
        momentum_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            step_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def rate_of_change(tacuda, json_data, res_index):
    """
    Calculate the rate of change for the given data.
    https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        rate_of_change_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def average_true_range(tacuda, json_data, res_index):
    """
    Calculate the average true range for the given data.
    https://en.wikipedia.org/wiki/Average_true_range

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        average_true_range_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )


def stochastic_oscillator_k(tacuda, json_data, res_index):
    """
    Calculate stochastic oscillator %K for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        stochastic_oscillator_k_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def stochastic_oscillator_d_ma(tacuda, json_data, res_index):
    """
    Calculate stochastic oscillator %D with moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        stochastic_oscillator_d_ma_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )


def stochastic_oscillator_d_ema(tacuda, json_data, res_index):
    """
    Calculate stochastic oscillator %D with exponential moving average for given data.
    https://en.wikipedia.org/wiki/Stochastic_oscillator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        stochastic_oscillator_d_ema_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )


def trix(tacuda, json_data, res_index):
    """
    Calculate TRIX for given data.
    https://en.wikipedia.org/wiki/Trix_(technical_analysis)

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        trix_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.result_gpu_mem,
            res_index + i
            )


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


def vortex_indicator_plus(tacuda, json_data, res_index):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        vortex_indicator_plus_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def vortex_indicator_minus(tacuda, json_data, res_index):
    """
    Calculate the Vortex Indicator for given data.
    https://en.wikipedia.org/wiki/Vortex_indicator

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        vortex_indicator_minus_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.result_gpu_mem,
            res_index + i
            )


def relative_strength_index(tacuda, json_data, res_index):
    """
    Calculate Relative Strength Index(RSI) for given data.
    https://en.wikipedia.org/wiki/Relative_strength_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        relative_strength_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.result_gpu_mem,
            res_index + i
            )


def true_strength_index(tacuda, json_data, res_index):
    """
    Calculate True Strength Index (TSI) for given data.
    https://en.wikipedia.org/wiki/True_strength_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
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


def money_flow_index(tacuda, json_data, res_index):
    """
    Calculate Money Flow Index and Ratio for given data.
    https://en.wikipedia.org/wiki/Money_flow_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        money_flow_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.temp_arr[1],
            tacuda.result_gpu_mem,
            res_index + i
            )

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


def force_index(tacuda, json_data, res_index):
    """
    Calculate Force Index for given data.
    https://en.wikipedia.org/wiki/Force_index

    :param tacuda: TACUDA
    :param json_data: function from config file
    :param res_index: column index in result array
    """
    for i, window_i in enumerate(json_data["windows"]):
        force_index_kernel_n[tacuda.blocks_per_grid, tacuda.threads_per_block](
            tacuda.ohlcv,
            window_i,
            tacuda.temp_arr[0],
            tacuda.result_gpu_mem,
            res_index + i
            )