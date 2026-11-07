"""Ordinary CTA-local split-K configurations and byte-address contracts."""

from dataclasses import dataclass

import torch
import triton

from flag_gems.runtime import torch_device_fn


def _loop_carried_refill_supported():
    try:
        from triton._C.libtriton import llvm
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return False
    return "loop_carried_store_refill" in HIPOptions.__dataclass_fields__ and callable(
        getattr(llvm, "interleave_hcu_loop_carried_store_refill", None)
    )


@dataclass(frozen=True)
class LocalSplitMMConfig:
    tile: tuple = (32, 32)
    k_step: int = 256
    # 0: ordinary, 1: peeled final tile, 2: input prefetch, 3: multiple row tiles.
    schedule: int = 0
    m_tiles: int = 1

    def __post_init__(self):
        object.__setattr__(self, "tile", tuple(self.tile))
        if len(self.tile) != 2 or any(
            type(x) is not int or x < 16 or x & (x - 1) for x in self.tile
        ):
            raise ValueError("local split tile dimensions must be powers of two >=16")
        bm, bn = self.tile
        if (
            type(self.k_step) is not int
            or self.k_step not in (128, 256)
            or type(self.schedule) is not int
            or self.schedule not in range(4)
        ):
            raise ValueError("unsupported local split step or schedule")
        if type(self.m_tiles) is not int or self.m_tiles < 1:
            raise ValueError("m_tiles must be a positive integer")
        if self.schedule != 3 and self.m_tiles != 1:
            raise ValueError("multiple row tiles require the multi-row schedule")
        if self.k_step == 128:
            if self.schedule or bm + bn > 128 or bm * bn > 4096:
                raise ValueError(
                    "K128 requires the ordinary schedule and a bounded shared tile"
                )
        elif self.m_tiles * bm + bn > 112 or (self.schedule != 3 and bm * bn > 4096):
            raise ValueError(
                "local split input planes exceed the 64KiB shared allocation"
            )

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
            and all(
                size == 1 or actual == expected
                for tensor, expected_strides in ((a, (k, 1)), (b, (1, k)), (c, (n, 1)))
                for size, actual, expected in zip(
                    tensor.shape, tensor.stride(), expected_strides
                )
            )
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and all(t.data_ptr() % 2 == 0 for t in (a, b, c))
            and disjoint
            # Grid/row masks use signed i32; allow their complete rounded tile.
            and m + self.m_tiles * bm < 2**31
            and n + bn < 2**31
            # Global A/B/C origins are rebased in i64. Only the within-CTA
            # buffer offsets, including inactive padded lanes, use signed i32.
            and 2 * ((max(bm, bn) - 1) * k + self.k_step) < 2**31
            and 2 * ((bm - 1) * n + bn) < 2**31
        )


_LOCAL_SPLIT_CONFIGS = (
    LocalSplitMMConfig((16, 32), k_step=128),
    LocalSplitMMConfig(),
    LocalSplitMMConfig(schedule=1),
    LocalSplitMMConfig((16, 16)),
    LocalSplitMMConfig((16, 16), schedule=1),
    LocalSplitMMConfig(schedule=2),
    LocalSplitMMConfig((16, 16), schedule=2),
    LocalSplitMMConfig((16, 32), schedule=3, m_tiles=5),
)


def local_split_mm_configs():
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return ()
    fields = HIPOptions.__dataclass_fields__
    if "padded_slot_intervals" not in fields:
        return ()
    return tuple(
        c
        for c in _LOCAL_SPLIT_CONFIGS
        if (
            c.k_step == 128
            or {"wave_private_sync", "disable_llvm_loop_unroll"} <= fields.keys()
        )
        and (c.schedule < 2 or _loop_carried_refill_supported())
    )


def launch_local_split_mm(a, b, c, config):
    if not isinstance(config, LocalSplitMMConfig) or not config.supports(a, b, c):
        raise ValueError("unsupported local split matrix storage or byte-offset range")
    from .mm_local_splitk_kernel import _mm_local_splitk_kernel

    m, k = a.shape
    n = b.shape[1]
    bm, bn = config.tile
    options = {
        "padded_slot_intervals": True,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 1,
        "loop_unroll_stage": "post-pipeline",
        "wave_private_sync": config.k_step == 256,
        "kpack": 2,
        "mmac_transposed": False,
    }
    if config.k_step == 128:
        options.update(
            INPUT_LAYOUT=2, PADDED=True, ROW_WARP_STRIDE=1, DOUBLE_BUFFER=True
        )
    else:
        options.update(LOOP_SCHEDULE=config.schedule, disable_llvm_loop_unroll=True)
        if config.m_tiles != 1:
            options["M_TILES"] = config.m_tiles
        if config.schedule >= 2:
            options["loop_carried_store_refill"] = True
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx936":
            raise ValueError("local split GEMM currently requires gfx936")
        return _mm_local_splitk_kernel[
            (triton.cdiv(m, bm * config.m_tiles), triton.cdiv(n, bn), 1)
        ](a, b, c, m, n, k, bm, bn, **options)
