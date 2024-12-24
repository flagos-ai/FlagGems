from dataclasses import dataclass
from typing import Tuple

import triton

from flag_gems.runtime import device
from flag_gems.runtime.commom_utils import vendors


@dataclass
class CodeGenConfig:
    max_tile_size: int
    max_grid_size: Tuple[int, int, int]
    max_num_warps_per_cta: int

    prefer_block_pointer: bool
    prefer_1d_tile: bool
    # gen_configs: -> configs
    # prune_config: (as jit function, ) cofigs -> configs

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
    )
}


def get_codegen_config():
    if device.vendor not in CODEGEN_COFIGS:
        return CODEGEN_COFIGS.get(vendors.NVIDIA)
    return CODEGEN_COFIGS.get(device.vendor)
