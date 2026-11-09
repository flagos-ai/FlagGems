"""Logical CTA-local split-K configurations for the ordinary Triton kernel."""

from dataclasses import dataclass

import torch
import triton

from flag_gems.runtime import torch_device_fn


def _matches_strides(tensor, expected):
    return all(
        size == 1 or actual == stride
        for size, actual, stride in zip(tensor.shape, tensor.stride(), expected)
    )


@dataclass(frozen=True)
class LocalSplitMMConfig:
    tile: tuple = (32, 32)
    k_step: int = 256
    m_tiles: int = 1
    num_stages: int = 1

    def __post_init__(self):
        object.__setattr__(self, "tile", tuple(self.tile))
        if len(self.tile) != 2 or any(
            type(x) is not int or x < 16 or x & (x - 1) for x in self.tile
        ):
            raise ValueError("local split tile dimensions must be powers of two >=16")
        if type(self.k_step) is not int or self.k_step not in (128, 256):
            raise ValueError("unsupported local split K step")
        if type(self.num_stages) is not int or self.num_stages not in (1, 2):
            raise ValueError("local split uses one or two pipeline stages")
        if type(self.m_tiles) is not int or self.m_tiles < 1:
            raise ValueError("m_tiles must be a positive integer")
        bm, bn = self.tile
        # Bound the simultaneous input planes and cross-wave FP32 output
        # scratch independently. Their lifetimes do not overlap.
        if max(2 * self.k_step * (self.m_tiles * bm + bn), 16 * bm * bn) > 65536:
            raise ValueError("local split tiles exceed the 64KiB shared allocation")

    def supports(self, a, b, c):
        if any(t.ndim != 2 for t in (a, b, c)):
            return False
        m, k = a.shape
        n = b.shape[1]
        bm, bn = self.tile
        begin, end = c.data_ptr(), c.data_ptr() + 2 * m * n
        disjoint = all(
            end <= t.data_ptr() or t.data_ptr() + 2 * count <= begin
            for t, count in ((a, m * k), (b, k * n))
        )
        return (
            min(m, n, k) > 0
            and b.shape[0] == k
            and c.shape == (m, n)
            and _matches_strides(a, (k, 1))
            and (_matches_strides(b, (1, k)) or _matches_strides(b, (n, 1)))
            and _matches_strides(c, (n, 1))
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and all(t.data_ptr() % 2 == 0 for t in (a, b, c))
            and disjoint
            # Coordinates and masks use signed i32, including inactive lanes.
            # All global pointer origins and element offsets use i64.
            and triton.cdiv(m, self.m_tiles * bm) * (self.m_tiles * bm) <= 2**31
            and triton.cdiv(n, bn) * bn <= 2**31
            and triton.cdiv(k, self.k_step) * self.k_step <= 2**31
        )


_LOCAL_SPLIT_CONFIGS = tuple(
    LocalSplitMMConfig(tile, k_step=step, m_tiles=rows, num_stages=stages)
    for stages in (1, 2)
    for tile, step, rows in (
        ((16, 32), 128, 1),
        ((32, 32), 256, 1),
        ((16, 16), 256, 1),
        ((16, 32), 256, 5),
    )
)


def local_split_mm_configs():
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return ()
    required = {"wave_private_sync", "kpack", "disable_llvm_loop_unroll"}
    return (
        _LOCAL_SPLIT_CONFIGS
        if required <= HIPOptions.__dataclass_fields__.keys()
        else ()
    )


def launch_local_split_mm(a, b, c, config):
    if not isinstance(config, LocalSplitMMConfig) or not config.supports(a, b, c):
        raise ValueError("unsupported local split matrix storage or coordinate range")
    from .mm_local_splitk_kernel import _mm_local_splitk_kernel

    m, k = a.shape
    n = b.shape[1]
    bm, bn = config.tile
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx936":
            raise ValueError("local split GEMM currently requires gfx936")
        return _mm_local_splitk_kernel[
            (triton.cdiv(m, bm * config.m_tiles), triton.cdiv(n, bn), 1)
        ](
            a,
            b,
            c,
            m,
            n,
            k,
            bm,
            bn,
            K_STEP=config.k_step,
            B_ROW_MAJOR=_matches_strides(b, (n, 1)),
            M_TILES=config.m_tiles,
            num_warps=4,
            num_stages=config.num_stages,
            waves_per_eu=1,
            schedule_hint="local-prefetch" if config.num_stages == 2 else "none",
            wave_private_sync=True,
            disable_llvm_loop_unroll=True,
            kpack=2,
            mmac_transposed=False,
        )
