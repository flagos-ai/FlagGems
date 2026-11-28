# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400-specific ``full``: a hand-written tl.full kernel replacing pointwise_dynamic codegen.

Production ``gcu400/ops/full.py`` goes through logger + check_dtype + torch.empty + the
pointwise_dynamic codegen full_func_scalar; almost all of a pure fill operator's time is spent in
the codegen Python layer. Here scalar fills of fp16/fp32/bf16 on GCU take a single Triton launch,
while other dtypes / Tensor fill_value / non-GCU devices all delegate to the production
implementation, keeping dtype policy and overflow-check semantics unchanged.

Launch shape (aligned with the same-family new_full three-round RSI calibration, GCU400 2026-11-16):
- grid capped at 24 (sip_count=24; the write path saturates at 24 CTAs; grid 48 is -3.7%, and the
  flat 16384-CTA form with a large contiguous span per block only reaches 2406GB/s in fp16/bf16);
- BLOCK tiers: n<=8192 uses one exact mask-free CTA block; larger takes
  min(8192, max(2048, np2(ceil(n/48)))) (B2048-B8192 are all tied at the optimum, B32768 degrades
  20-64% in the 16K-64K element tier and 14% at 2^20 fp32, and a single np2(n) block of
  >=2^20 elements triggers DSM OOM), with a grid-stride loop reusing programs in-kernel;
- num_warps=2 (nw4 is -2% on the large tier); the large 2^30 tier hits the write wall at
  3246-3262GB/s for every BLOCK (DRAM write wall 3.25/3.26TB/s), while the 2^24 mid band gives 2581-2900GB/s.

The pin_memory guard is truthy-based (only True delegates to production): the ATen registration
chain passes ``pin_memory=False`` (not None) to this function, so a strict ``is not None`` test
would make the GCU fast path unreachable under torch.full/ATen dispatch, degrading to the
production codegen version (topsprof shows full_func_scalar at 702us, the real body behind the
QA-reported 1.27x); False and None are semantically equivalent (torch.empty pins in neither case),
so only pin_memory=True delegates to the production implementation.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.ops.full import full as _public
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["full"]

logger = logging.getLogger(__name__)

_FLAT_MAX_GRID = 24
_FLAT_CAP_BLOCK = 8192
_FLAT_MIN_BLOCK = 2048
_TINY_MAX_ELEMS = 8192
_FLAT_MAX_ELEMS = 2**31 - 1  # int32 offsets guard, larger falls back to _public

_FINFO = {
    torch.float16: torch.finfo(torch.float16),
    torch.float32: torch.finfo(torch.float32),
    torch.bfloat16: torch.finfo(torch.bfloat16),
}


@libentry()
@triton.jit
def _full_kernel(
    OUT,
    N,
    VAL,
    BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """Each program produces a constant vector; with FIXED_GRID>0 programs are reused by in-kernel striding."""
    arange = tl.arange(0, BLOCK_SIZE)
    values = tl.full([BLOCK_SIZE], VAL, OUT.dtype.element_ty)
    if FIXED_GRID > 0:
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for block_id in tl.range(pid, tl.cdiv(N, BLOCK_SIZE), num_programs):
            offs = block_id * BLOCK_SIZE + arange
            if MASKED:
                tl.store(OUT + offs, values, mask=offs < N)
            else:
                tl.store(OUT + offs, values)
    else:
        offs = tl.program_id(0) * BLOCK_SIZE + arange
        if MASKED:
            tl.store(OUT + offs, values, mask=offs < N)
        else:
            tl.store(OUT + offs, values)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME FULL")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dtype is None:
        if isinstance(fill_value, bool):
            dt = torch.bool
        elif isinstance(fill_value, int):
            dt = torch.int64
        else:
            dt = torch.get_default_dtype()
    else:
        dt = dtype

    if (
        dev.type != "gcu"
        or dt not in _FINFO
        or isinstance(fill_value, torch.Tensor)
        or layout is not None
        or (pin_memory is not None and pin_memory)
    ):
        return _public(
            size,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    # Same semantics as production check_dtype: bool scalars under a floating dtype are cast to int first, then
    # overflow-checked.
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
    if not (math.isinf(fill_value) or math.isnan(fill_value)):
        finfo = _FINFO[dt]
        if fill_value < finfo.min or fill_value > finfo.max:
            raise RuntimeError(
                f"value cannot be converted to type {dt} without overflow"
            )

    out = torch.empty(size, device=dev, dtype=dt)
    num_elems = out.numel()
    if num_elems == 0:
        return out
    if num_elems > _FLAT_MAX_ELEMS:
        # int32 offsets cannot address larger fills; keep semantics exact.
        return _public(
            size,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    if num_elems <= _TINY_MAX_ELEMS:
        block = max(2, triton.next_power_of_2(num_elems))
    else:
        block = min(
            _FLAT_CAP_BLOCK,
            max(
                _FLAT_MIN_BLOCK,
                triton.next_power_of_2(-(-num_elems // _FLAT_MAX_GRID)),
            ),
        )
    grid = (min(triton.cdiv(num_elems, block), _FLAT_MAX_GRID),)
    _full_kernel[grid](
        out,
        num_elems,
        fill_value,
        BLOCK_SIZE=block,
        MASKED=(num_elems % block != 0) or (block > num_elems),
        FIXED_GRID=_FLAT_MAX_GRID,
        num_warps=2,
        num_stages=1,
    )
    return out
