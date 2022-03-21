import math
from numba import cuda, float32, int32, jit
import os
import numpy as np
from .ta_functions import *

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

    
    def process(self):
        res_index = 0
        for i, function_i in enumerate(self.json_data["technical indicators"]):
            ta_function = function_i["function"]
            ta_params = {
                "tacuda": self,
                "json_data": function_i,
                "res_index": res_index
            }
            globals()[ta_function](**ta_params)
            res_index += max(1, len(function_i["windows"]))

        
