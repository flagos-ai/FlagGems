from dataclasses import dataclass
from typing import Tuple

import triton

from flag_gems.runtime import device
from flag_gems.runtime.backend import vendor_module
from flag_gems.runtime.common import vendors


def default_heuristics_for_num_warps(tile_size):
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def metax_heuristics_for_num_warps(tile_size):
    if tile_size <= 1024:
        return 4
    elif tile_size <= 2048:
        return 8
    else:
        return 16


def cambricon_heuristics_for_num_warps(tile_size):
    return 1


@dataclass
class CodeGenConfig:
    max_tile_size: int
    max_grid_size: Tuple[int, int, int]
    max_num_warps_per_cta: int

    prefer_block_pointer: bool
    prefer_1d_tile: bool
    # gen_configs: -> configs
    # prune_config: (as jit function, ) cofigs -> configs
    is_scatter_slice: bool = False
    is_cat: bool = False
    isCloseVectorization: bool = False
    isCloseDtypeConvert: bool = False
    isCloseMemoryAsync: bool = True
    isCloseInterleave: bool = False
    kunlunAutoGrid: bool = False
    buffer_size_limit: int = 0
    unroll_num: int = 0

    def __post_init__(self):
        if self.prefer_1d_tile:
            self.prefer_block_pointer = False


CODEGEN_COFIGS = {
    vendors.NVIDIA: CodeGenConfig(
        512,
        (65536, 65536, 65536),
        32,
        True,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    ),
    vendors.CAMBRICON: CodeGenConfig(
        8192,
        tuple([vendor_module.TOTAL_CORE_NUM, 1, 1]),
        32,
        False,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    )
    if vendor_module.vendor_info.vendor_name == "cambricon"
    else None,
    vendors.METAX: CodeGenConfig(
        2048,
        (65536, 65536, 65536),
        16,
        True,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    ),
    vendors.KUNLUNXIN: CodeGenConfig(
        512,
        (65536, 65536, 65536),
        32,
        True,
        prefer_1d_tile=True,
    ),
}

HEURISTICS_CONFIG = {
    vendors.NVIDIA: default_heuristics_for_num_warps,
    vendors.METAX: metax_heuristics_for_num_warps,
    vendors.CAMBRICON: cambricon_heuristics_for_num_warps,
}


def get_codegen_config():
    if device.vendor not in CODEGEN_COFIGS:
        return CODEGEN_COFIGS.get(vendors.NVIDIA)
    return CODEGEN_COFIGS.get(device.vendor)


def get_heuristics_for_num_warps(tile_size):
    if device.vendor not in HEURISTICS_CONFIG:
        return HEURISTICS_CONFIG.get(vendors.NVIDIA)(tile_size)
    return HEURISTICS_CONFIG.get(device.vendor)(tile_size)
