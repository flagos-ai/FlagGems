"""Explicit geometry choices for the ordinary TL/TLE GEMM entry."""

from dataclasses import dataclass
from functools import cache, lru_cache
from math import prod

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@dataclass(frozen=True)
class SharedMMConfig:
    tile: tuple
    warps: tuple = (2, 2)
    group_m: int = 8
    address_mode: str = "direct"

    def __post_init__(self):
        if len(self.tile) != 3 or any(x < 8 or x & (x - 1) for x in self.tile):
            raise ValueError("tile dimensions must be powers of two >= 8")
        bm, bn, bk = self.tile
        if max(bm * bk, bn * bk) > 4096:
            raise ValueError("operand tiles must fit the shared slot")
        if (
            len(self.warps) != 2
            or any(x < 1 or x & (x - 1) for x in self.warps)
            or prod(self.warps) > 8
        ):
            raise ValueError("requires power-of-two wave axes and at most 8 waves")
        if self.group_m < 1:
            raise ValueError("traversal group must be positive")
        if self.address_mode not in ("direct", "fold", "scalar"):
            raise ValueError("unknown buffer-address strategy")

    def supports(self, a, b, c):
        """Check vector alignment, storage disjointness and address bounds."""
        if any(t.ndim != 2 for t in (a, b, c)):
            return False
        m, k = a.shape
        n = b.shape[1]
        bm, bn, bk = self.tile
        nt = b.stride() == (1, k)
        local_span = max(
            (bm - 1) * k + bk, (bn - 1) * k + bk if nt else (bk - 1) * n + bn
        )
        output_begin, output_end = c.data_ptr(), c.data_ptr() + 2 * m * n
        disjoint = all(
            output_end <= t.data_ptr() or t.data_ptr() + 2 * count <= output_begin
            for t, count in ((a, m * k), (b, k * n))
        )
        return (
            m > 0
            and n > 0
            and k >= 8
            and k % 8 == 0
            and (nt or n % 8 == 0)
            and b.shape[0] == k
            and tuple(c.shape) == (m, n)
            and a.stride() == (k, 1)
            and (nt or b.stride() == (n, 1))
            and c.stride() == (n, 1)
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and disjoint
            and max(m, n, k) < 2**31
            and 2 * m * n < 2**31
            and (2 * local_span < 2**31 or max(2 * m * k, 2 * k * n) < 2**31)
        )


@cache
def _compile_parameters(config, transposed_b):
    options = {
        "num_warps": prod(config.warps),
        "num_stages": 5,
        "waves_per_eu": 1,
        "loop_unroll_stage": "post-pipeline",
        "schedule_hint": "local-prefetch",
        "lds_dma_alias_workaround": True,
        "prefer_wide_async_copy": True,
        "kpack": 2,
        "mmac_warps_m": config.warps[0],
        "mmac_transposed": True,
        "optimize_epilogue": True,
        "rebase_buffer_store": False,
        "terminal_readonly_publication": False,
        "GROUP_M": config.group_m,
        "BLOCK_M": config.tile[0],
        "BLOCK_N": config.tile[1],
        "BLOCK_K": config.tile[2],
        "USE_TLE": True,
        "ASYNC_LOAD": True,
        "ASYNC_LOAD_MASK": 3,
        "LOAD_MODE": 3,
        "LARGE_TILE_CONFIGS": True,
        "LOOP_UNROLL_FACTOR": 4,
        "B_K_CONTIGUOUS": transposed_b,
    }
    if config.address_mode != "direct":
        options.update(
            coalesce_lgkm_waits=True,
            combine_shifted_disjoint_or=True,
            buffer_store_scalar_offset=config.address_mode == "scalar",
            buffer_store_split_immediate=True,
            buffer_store_preserve_masked_base=True,
        )
    # Cache immutable items; callers never share a mutable options dictionary.
    return tuple(options.items())


@lru_cache(maxsize=1)
def _ordinary_mm_entry():
    # Resolve lazily because mm imports its family dispatcher. Configurations
    # are selected by that dispatcher; do not nest another autotuner here.
    from .mm import mm_kernel

    kernel = mm_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    return libentry()(kernel)


def launch_shared_mm(a, b, c, config):
    """Launch one explicit ordinary GEMM candidate through the real LibEntry."""
    if not config.supports(a, b, c):
        raise ValueError("unsupported matrix storage, vector extent or address range")
    m, k = a.shape
    n = b.shape[1]
    bm, bn, _ = config.tile
    tiles_m, tiles_n = triton.cdiv(m, bm), triton.cdiv(n, bn)
    padded_m, padded_n = tiles_m * bm, tiles_n * bn
    options = dict(_compile_parameters(config, b.stride() == (1, k)))
    options.update(
        MASK_EDGES=m % bm != 0 or n % bn != 0,
        K_DIVISIBLE_BY_16=k % 16 == 0,
        # Only tile coordinates and output offsets are narrowed. Block-pointer
        # shapes/strides and global input origins retain their i64 arithmetic.
        USE_I32_INDEX=max(
            tiles_m * tiles_n,
            config.group_m * tiles_n,
            padded_m,
            padded_n,
            2 * ((padded_m - 1) * n + padded_n),
        )
        < 2**31,
    )
    with torch_device_fn.device(a.device):
        return _ordinary_mm_entry()[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            a,
            b,
            c,
            tl.constexpr(m),
            tl.constexpr(n),
            tl.constexpr(k),
            *(tl.constexpr(s) for s in a.stride()),
            *(tl.constexpr(s) for s in b.stride()),
            *(tl.constexpr(s) for s in c.stride()),
            **options,
        )


@lru_cache(maxsize=1)
def shared_mm_configs():
    """Logical tiles and traversal choices, shared by NN and NT.

    Lane ownership, shared layout, copy instruction and synchronization are
    selected by the compiler. Do not duplicate candidates for those details.
    """
    return (
        SharedMMConfig((64, 128, 32)),
        SharedMMConfig((32, 64, 64)),
        SharedMMConfig((128, 128, 32)),
        SharedMMConfig((128, 256, 16)),
        SharedMMConfig((64, 64, 64)),
        SharedMMConfig((256, 256, 16), warps=(4, 2), group_m=1, address_mode="scalar"),
        SharedMMConfig((256, 256, 16), warps=(4, 2), address_mode="fold"),
        SharedMMConfig((256, 256, 16), warps=(4, 2), group_m=1, address_mode="fold"),
        SharedMMConfig((128, 128, 16), address_mode="scalar"),
        SharedMMConfig((256, 256, 16), warps=(4, 2)),
        SharedMMConfig((128, 256, 16), address_mode="scalar"),
        SharedMMConfig((64, 64, 64), group_m=1),
    )
