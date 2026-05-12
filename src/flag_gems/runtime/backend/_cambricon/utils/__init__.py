import sys

import torch
import triton.backends.mlu.driver as driver

_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())

TOTAL_CLUSTER_NUM = _devprob.get("cluster_num")
TOTAL_CORE_NUM = TOTAL_CLUSTER_NUM * _devprob.get("core_num_per_cluster")
MAX_NRAM_SIZE = _devprob.get("max_nram_size")
DEVICE_COUNT = torch.mlu.device_count()
MAX_GRID_SIZES = [
    _devprob.get("max_block_task_dim_x", sys.maxsize),
    _devprob.get("max_block_task_dim_y", sys.maxsize),
    _devprob.get("max_block_task_dim_z", sys.maxsize),
]
MAX_GRID_SIZE_X, MAX_GRID_SIZE_Y, MAX_GRID_SIZE_Z = MAX_GRID_SIZES

from .reduce_utils import *  # noqa F403 F401

# from .pointwise_dynamic import pointwise_dynamic

__all__ = [
    "TOTAL_CORE_NUM",
    "TOTAL_CLUSTER_NUM",
    "MAX_NRAM_SIZE",
    "MAX_GRID_SIZE_X",
    "MAX_GRID_SIZE_Y",
    "MAX_GRID_SIZE_Z",
    "MAX_GRID_SIZES",
    "DEVICE_COUNT",
]
