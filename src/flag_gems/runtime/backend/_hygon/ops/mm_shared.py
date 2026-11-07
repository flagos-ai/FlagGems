"""Ordinary shared-layout GEMM launch policy; no Packet types or source ABI."""

from dataclasses import dataclass
from functools import lru_cache
from math import prod

import torch
import triton

from flag_gems.runtime import torch_device_fn

from .mm_shared_layout import operand_shared_layouts


@dataclass(frozen=True)
class SharedMMConfig:
    tile: tuple
    warps: tuple = (2, 2)
    tiles_per_warp: tuple = (2, 2)
    group_m: int = 8
    output_layout: int = 0
    row_group_bits: int = 2
    matrix_copy: bool = True
    post_mma: bool = False
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
        if self.matrix_copy and min(bm * bk, bn * bk) < 512 * prod(self.warps):
            raise ValueError("matrix copy requires a complete 16B vector per lane/wave")
        if len(self.tiles_per_warp) != 2 or any(
            x < 1 or x & (x - 1) for x in self.tiles_per_warp
        ):
            raise ValueError("MMAC tiles per wave axis must be positive powers of two")
        if self.group_m < 1 or self.output_layout not in (0, 1, 2):
            raise ValueError("invalid traversal or output layout")
        if self.address_mode not in ("direct", "fold", "scalar"):
            raise ValueError("unknown buffer-address strategy")
        if not 0 <= self.row_group_bits <= 6:
            raise ValueError("invalid row grouping")
        if self.output_layout == 2 and bn < 32:
            raise ValueError("column ownership requires a 32-column tile")

    def supports(self, a, b, c):
        """Check actual vector, pipeline and byte-offset requirements."""
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
            and k >= 4 * bk
            and k % (4 * bk) == 0
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
            and (
                2 * local_span < 2**31
                if self.matrix_copy
                else max(2 * m * k, 2 * k * n) < 2**31
            )
        )


@lru_cache(maxsize=None)
def _compile_parameters(config, transposed_b):
    a_layout, b_layout = operand_shared_layouts(
        config.tile, transposed_b=transposed_b, row_group_bits=config.row_group_bits
    )
    options = dict(
        num_warps=prod(config.warps),
        num_stages=5,
        waves_per_eu=1,
        loop_unroll_stage="post-pipeline",
        schedule_hint="local-prefetch",
        lds_dma_alias_workaround=True,
        mma_phase_schedule=config.post_mma,
        post_mma_wait=config.post_mma,
        rebase_buffer_store=False,
        terminal_readonly_publication=False,
        GROUP_M=config.group_m,
        TILES_PER_WARP=config.tiles_per_warp,
        OUTPUT_LAYOUT=config.output_layout,
        A_SHARED_LAYOUT=a_layout,
        B_SHARED_LAYOUT=b_layout,
        USE_MATRIX_COPY=config.matrix_copy,
    )
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


def launch_shared_mm(a, b, c, config):
    """Launch one explicit ordinary GEMM candidate through the real LibEntry."""
    if not config.supports(a, b, c):
        raise ValueError("unsupported matrix storage, vector extent or pipeline K")
    from .mm_shared_kernel import _mm_shared_kernel

    m, k = a.shape
    n = b.shape[1]
    bm, bn, _ = config.tile
    options = dict(_compile_parameters(config, b.stride() == (1, k)))
    with torch_device_fn.device(a.device):
        return _mm_shared_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            a,
            b,
            c,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            *c.stride(),
            *config.tile,
            *config.warps,
            **options,
        )


@lru_cache(maxsize=1)
def shared_mm_configs():
    """One geometry registry shared by both runtime B storage layouts.

    As with the ordinary YAML pool, a configuration is not duplicated for its
    input strides or dtype. All entries are offered for NN and NT; only actual
    vector/resource/pipeline legality is checked against the current operands.
    """
    from dataclasses import replace

    geometries = (
        SharedMMConfig((64, 128, 32)),
        SharedMMConfig((32, 64, 64), tiles_per_warp=(1, 2)),
        SharedMMConfig((128, 128, 32)),
        SharedMMConfig((128, 256, 16)),
        SharedMMConfig((64, 64, 64), tiles_per_warp=(1, 1)),
        SharedMMConfig((64, 64, 64)),
        SharedMMConfig((256, 256, 16), warps=(4, 2), group_m=1, address_mode="scalar"),
        SharedMMConfig(
            (256, 256, 16),
            warps=(4, 2),
            row_group_bits=3,
            output_layout=2,
            address_mode="fold",
        ),
        SharedMMConfig(
            (256, 256, 16),
            warps=(4, 2),
            row_group_bits=3,
            output_layout=2,
            group_m=1,
            address_mode="fold",
        ),
    )
    regular = tuple(
        replace(geometry, matrix_copy=copy, post_mma=post)
        for geometry in geometries
        for copy in (True, False)
        for post in (False, True)
    )
    # Both operands fit a half-size allocation frame. Keep this useful tile
    # geometry in the same ordinary registry instead of a compact-owner ABI.
    return regular + (
        SharedMMConfig(
            (128, 128, 16),
            output_layout=1,
            address_mode="scalar",
            matrix_copy=False,
            post_mma=True,
        ),
        SharedMMConfig(
            (256, 256, 16), warps=(4, 2), row_group_bits=3, post_mma=True
        ),
        SharedMMConfig((128, 256, 16), output_layout=2, address_mode="scalar"),
        SharedMMConfig((64, 64, 64), group_m=1, post_mma=True),
    )
