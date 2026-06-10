"""Synthetic kernel for testing autotune_bridge.

Mirrors the bmm-like shape (positional args + autotune kwargs + heuristics
kwargs + a defaulted constexpr) without needing a GPU, since:
  - `triton.jit` is lazy (no compile at import time)
  - LibTuner with a single config short-circuits resolve_config (no benchmark)
  - Heuristics are pure-Python functions
"""

import triton
import triton.language as tl

from flag_gems.utils.libentry import libentry, libtuner


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"TILE_M": 128, "TILE_N": 64, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    {
        "DIVISIBLE_M": lambda nargs: nargs["M"] % nargs["TILE_M"] == 0,
        "DIVISIBLE_N": lambda nargs: nargs["N"] % nargs["TILE_N"] == 0,
    }
)
@triton.jit
def synth_kernel(
    A,  # tensor (non-constexpr)
    B,  # tensor (non-constexpr)
    M,  # int (non-constexpr)
    N,  # int (non-constexpr)
    TILE_M: tl.constexpr,  # from autotune
    TILE_N: tl.constexpr,  # from autotune
    GROUP_M: tl.constexpr,  # from autotune
    DIVISIBLE_M: tl.constexpr,  # from heuristics
    DIVISIBLE_N: tl.constexpr,  # from heuristics
    IS_FP64: tl.constexpr = False,  # from default
):
    pass
