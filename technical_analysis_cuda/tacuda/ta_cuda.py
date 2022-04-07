import math
from numba import cuda, float32, int32, jit
import os
import numpy as np
#from .ta_functions import *
from .ta_kernels import *

class TACUDA:
    def __init__(self, timeline, json_data, threads_per_block=32):
        self.json_data = json_data
        self.ohlcv = cuda.to_device(timeline)
        self.tech_inds_n = self.tech_inds_num(json_data)
        self.temp_arr = [cuda.device_array(shape=timeline.shape[0], dtype=np.float64) for i in range(4)]
        self.result_gpu_mem = cuda.device_array(shape=(timeline.shape[0], self.tech_inds_n), dtype=np.float64)
        self.threads_per_block = threads_per_block
        self.blocks_per_grid = self.cuda_blocks_per_grid(timeline.shape[0], threads_per_block)
    
    def tech_inds_num(self, json_data):
        tech_inds_n = 0
        for function_i in json_data["technical indicators"]:
            tech_inds_n += len(function_i["windows"]) if len(function_i["windows"]) > 0 else 1
        return tech_inds_n

    ##################################
    # CUDA preparation functions
    ##################################

    def cuda_blocks_per_grid(self, length, threads_per_block):
        """
        Calculate the number of cuda blocks per grid for length of an array
        
        :param length: lenght of the array
        :param threads_per_block: the number of threads per block
        """
        
        return math.ceil(length / threads_per_block)


    def cuda_blocks_per_grid_2d(self, shape, threads_per_block_2d):
        """
        Calculate the number of cuda blocks per grid for length of an 2d array

        :param shape: shape of the 2d array
        :param threads_per_block: the number of threads per block
        """
        return (int(math.ceil(shape[0] / threads_per_block_2d[0])),
                int(math.ceil(shape[1] / threads_per_block_2d[1])))


    def cuda_device_info(self):
        """
        Print cuda and GPU information
        """
        os.system('nvcc --version')
        os.system('nvidia-smi')
        print(cuda.detect())

    ##################################
    # Process function
    ##################################
    
    def process(self):
        res_index = 0
        names = []
        for i, function_i in enumerate(self.json_data["technical indicators"]):
            ta_function = function_i["function"]
            ta_params = {
                "json_data": function_i,
                "res_index": res_index
            }
            names += getattr(self, ta_function)(**ta_params)
            res_index += max(1, len(function_i["windows"]))
        return names
    

    ##################################
    # Technical analysis function
    ##################################


    def moving_average(self, json_data, res_index):
        """
        Calculate the moving average for the given data.
        https://en.wikipedia.org/wiki/Moving_average

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            moving_average_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["moving_average" + "." + str(window_i)]
        return names


    def exponential_moving_average(self, json_data, res_index):
        """
        Calculate the exponential moving average for the given data.
        https://en.wikipedia.org/wiki/Moving_average

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            exponential_moving_average_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["exponential_moving_average" + "." + str(window_i)]
        return names


    def momentum(self, json_data, res_index):
        """
        Calculate the momentum for the given data.
        https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, step_i in enumerate(json_data["windows"]):
            momentum_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                step_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["momentum" + "." + str(step_i)]
        return names


    def rate_of_change(self, json_data, res_index):
        """
        Calculate the rate of change for the given data.
        https://en.wikipedia.org/wiki/Momentum_(technical_analysis)

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            rate_of_change_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["rate_of_change" + "." + str(window_i)]
        return names


    def average_true_range(self, json_data, res_index):
        """
        Calculate the average true range for the given data.
        https://en.wikipedia.org/wiki/Average_true_range

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            average_true_range_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["average_true_range" + "." + str(window_i)]
        return names


    def stochastic_oscillator_k(self, json_data, res_index):
        """
        Calculate stochastic oscillator %K for given data.
        https://en.wikipedia.org/wiki/Stochastic_oscillator

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            stochastic_oscillator_k_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["stochastic_oscillator_k" + "." + str(window_i)]
        return names


    def stochastic_oscillator_d_ma(self, json_data, res_index):
        """
        Calculate stochastic oscillator %D with moving average for given data.
        https://en.wikipedia.org/wiki/Stochastic_oscillator

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            stochastic_oscillator_d_ma_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["stochastic_oscillator_d_ma" + "." + str(window_i)]
        return names


    def stochastic_oscillator_d_ema(self, json_data, res_index):
        """
        Calculate stochastic oscillator %D with exponential moving average for given data.
        https://en.wikipedia.org/wiki/Stochastic_oscillator

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            stochastic_oscillator_d_ema_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["stochastic_oscillator_d_ema" + "." + str(window_i)]
        return names


    def trix(self, json_data, res_index):
        """
        Calculate TRIX for given data.
        https://en.wikipedia.org/wiki/Trix_(technical_analysis)

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            trix_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.temp_arr[1],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["trix" + "." + str(window_i)]
        return names


    def mass_index(self, json_data, res_index):
        """
        Calculate the Mass Index for given data.
        https://en.wikipedia.org/wiki/Mass_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        mass_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
            self.ohlcv,
            json_data["param"],
            self.temp_arr[0],
            self.temp_arr[1],
            self.result_gpu_mem,
            res_index
            )
        return ["mass_index" + "." + str(json_data["param"])]


    def vortex_indicator_plus(self, json_data, res_index):
        """
        Calculate the Vortex Indicator for given data.
        https://en.wikipedia.org/wiki/Vortex_indicator

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            vortex_indicator_plus_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["vortex_indicator_plus" + "." + str(window_i)]
        return names


    def vortex_indicator_minus(self, json_data, res_index):
        """
        Calculate the Vortex Indicator for given data.
        https://en.wikipedia.org/wiki/Vortex_indicator

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            vortex_indicator_minus_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["vortex_indicator_minus" + "." + str(window_i)]
        return names


    def relative_strength_index(self, json_data, res_index):
        """
        Calculate Relative Strength Index(RSI) for given data.
        https://en.wikipedia.org/wiki/Relative_strength_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            relative_strength_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.temp_arr[1],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["relative_strength_index" + "." + str(window_i)]
        return names


    def true_strength_index(self, json_data, res_index):
        """
        Calculate True Strength Index (TSI) for given data.
        https://en.wikipedia.org/wiki/True_strength_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, (window_1, window_2) in enumerate(zip(json_data["windows"], json_data["windows_2"])):
            true_strength_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_1,
                window_2,
                self.temp_arr[0],
                self.temp_arr[1],
                self.temp_arr[2],
                self.temp_arr[3],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["true_strength_index" + "." + str(json_data["windows"][i]) + "." + str(json_data["windows_2"][i])]
        return names


    def accumulation_distribution(self, json_data, res_index):
        """
        Calculate Accumulation/Distribution for given data.
        https://en.wikipedia.org/wiki/Accumulation/distribution_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        accumulation_distribution_kernel_n[self.blocks_per_grid, self.threads_per_block](
            self.ohlcv,
            self.temp_arr[0],
            self.result_gpu_mem,
            res_index
            )
        return ["accumulation_distribution"]


    def chaikin_oscillator(self, json_data, res_index):
        """
        Calculate Chaikin Oscillator for given data.
        https://en.wikipedia.org/wiki/Chaikin_Analytics

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        chaikin_oscillator_kernel_n[self.blocks_per_grid, self.threads_per_block](
            self.ohlcv,
            self.temp_arr[0],
            self.temp_arr[1],
            self.temp_arr[2],
            self.result_gpu_mem,
            res_index
            )
        return ["chaikin_oscillator"]


    def chaikin_money_flow(self, json_data, res_index):
        """
        Calculate Money Flow for given data.
        https://en.wikipedia.org/wiki/Chaikin_Analytics

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        chaikin_money_flow_kernel_n[self.blocks_per_grid, self.threads_per_block](
            self.ohlcv,
            self.temp_arr[0],
            self.result_gpu_mem,
            res_index
            )
        return ["chaikin_money_flow"]


    def money_flow_index(self, json_data, res_index):
        """
        Calculate Money Flow Index and Ratio for given data.
        https://en.wikipedia.org/wiki/Money_flow_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            money_flow_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.temp_arr[1],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["money_flow_index" + "." + str(window_i)]
        return names

    def on_balance_volume(self, json_data, res_index):
        """
        Calculate On-Balance Volume for given data.
        https://en.wikipedia.org/wiki/On-balance_volume

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        on_balance_volume_kernel_n[self.blocks_per_grid, self.threads_per_block](
            self.ohlcv,
            self.result_gpu_mem,
            res_index
            )
        return ["on_balance_volume"]


    def force_index(self, json_data, res_index):
        """
        Calculate Force Index for given data.
        https://en.wikipedia.org/wiki/Force_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            force_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["force_index" + "." + str(window_i)]
        return names


    def ease_of_movement(self, json_data, res_index):
        """
        Calculate Ease of Movement for given data.
        https://en.wikipedia.org/wiki/Ease_of_movement

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            ease_of_movement_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["ease_of_movement" + "." + str(window_i)]
        return names


    def standard_deviation(self, json_data, res_index):
        """
        Calculate Standard Deviation for given data.
        https://en.wikipedia.org/wiki/Standard_deviation

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            standard_deviation_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["standard_deviation" + "." + str(window_i)]
        return names
        

    def commodity_channel_index(self, json_data, res_index):
        """
        Calculate Commodity Channel Index for given data.
        https://en.wikipedia.org/wiki/Commodity_channel_index

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            commodity_channel_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.temp_arr[0],
                self.result_gpu_mem,
                res_index + i
                )
            names += ["commodity_channel_index" + "." + str(window_i)]
        return names


    def keltner_channel_m(self, json_data, res_index):
        """
        Calculate Keltner Channel Middle for given data.
        https://www.investopedia.com/terms/k/keltnerchannel.asp

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            keltner_channel_m_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["keltner_channel_m" + "." + str(window_i)]
        return names


    def keltner_channel_u(self, json_data, res_index):
        """
        Calculate Keltner Channel Up for given data.
        https://www.investopedia.com/terms/k/keltnerchannel.asp

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            keltner_channel_u_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["keltner_channel_u" + "." + str(window_i)]
        return names


    def keltner_channel_d(self, json_data, res_index):
        """
        Calculate Keltner Channel Down for given data.
        https://www.investopedia.com/terms/k/keltnerchannel.asp

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            keltner_channel_d_index_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["keltner_channel_d" + "." + str(window_i)]
        return names

    def ultimate_oscillator(self, json_data, res_index):
        """
        Calculate Ultimate Oscillator for given data.
        https://en.wikipedia.org/wiki/Ultimate_oscillator

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        ultimate_oscillator_kernel_n[self.blocks_per_grid, self.threads_per_block](
            self.ohlcv,
            self.temp_arr[0],
            self.temp_arr[1],
            self.result_gpu_mem,
            res_index
            )
        return ["ultimate_oscillator"]


    def donchian_channel_u(self, json_data, res_index):
        """
        Calculate Donchian Channel Up of given data.
        https://www.investopedia.com/terms/d/donchianchannels.asp

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            donchian_channel_u_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["donchian_channel_u" + "." + str(window_i)]
        return names


    def donchian_channel_d(self, json_data, res_index):
        """
        Calculate Donchian Channel Down of given data.
        https://www.investopedia.com/terms/d/donchianchannels.asp

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            donchian_channel_d_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["donchian_channel_d" + "." + str(window_i)]
        return names


    def donchian_channel_m(self, json_data, res_index):
        """
        Calculate Donchian Channel Middle of given data.
        https://www.investopedia.com/terms/d/donchianchannels.asp

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            donchian_channel_m_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["donchian_channel_m" + "." + str(window_i)]
        return names


    def bollinger_bands_m(self, json_data, res_index):
        """
        Calculate Bollinger bands Middle for the given data.
        https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            bollinger_bands_m_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["bollinger_bands_m" + "." + str(window_i)]
        return names


    def bollinger_bands_u(self, json_data, res_index):
        """
        Calculate Bollinger bands Up for the given data.
        https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            bollinger_bands_u_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["bollinger_bands_u" + "." + str(window_i)]
        return names
            


    def bollinger_bands_d(self, json_data, res_index):
        """
        Calculate Bollinger bands Down for the given data.
        https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/

        :param json_data: function from config file
        :param res_index: column index in result array
        """
        names = []
        for i, window_i in enumerate(json_data["windows"]):
            bollinger_bands_d_kernel_n[self.blocks_per_grid, self.threads_per_block](
                self.ohlcv,
                window_i,
                self.result_gpu_mem,
                res_index + i
                )
            names += ["bollinger_bands_d" + "." + str(window_i)]
        return names
            
