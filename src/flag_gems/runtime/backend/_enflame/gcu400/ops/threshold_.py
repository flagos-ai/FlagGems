# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# GCU400 flat-kernel override of the pointwise_dynamic path (same recipe as
# threshold.py, 2026-11-11).  The in-place pointwise_dynamic path (out0=A)
# is far worse than the out-of-place one: ~0.35 TB/s flat across dtypes
# (16M 182-185us vs 28us flat, 1G 11.5ms vs 1.35ms native, geomean 0.25x).
# The flat grid-stride kernel below writes back to A elementwise (load ->
# where -> store on the same offsets, no cross-block overlap), keeping the
# identical math (fp32-upcast compare, bit-exact vs the pointwise kernel).
# Small / non-contiguous / non-float / huge (> int32-safe) inputs keep the
# generic pointwise path, which is fastest below ~8K elements.
NUM_SIPS = 24
_SMALL_N = 8192
_SINGLE_CTA_N = 32768
_MID_N = 1 << 21
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def threshold__flat_kernel(
    x_ptr, out_ptr, N_total, threshold, value, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        m = off < N_total
        x = tl.load(x_ptr + off, mask=m)
        y = tl.where(x > threshold, x, value).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + off, y, mask=m)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def threshold_func(x, threshold, value):
    # Perform computation in float32 for precision
    x_fp32 = x.to(tl.float32)
    threshold_fp32 = threshold.to(tl.float32)
    value_fp32 = value.to(tl.float32)
    # If x <= threshold, return value, otherwise return x
    return tl.where(x_fp32 <= threshold_fp32, value_fp32, x_fp32).to(x.dtype)


def _choose_block(N_total):
    # do_bench(median) sweep on GCU400 (threshold.py, same hardware): mid
    # shapes peak at BLOCK 32768, the 16M-1G band gains ~1.5-3% from 65536;
    # tiny shapes use one exact next_pow2 tile.  131072 was rejected: no
    # gain and it exceeds the 917504B DSM budget.
    if N_total <= _SINGLE_CTA_N:
        return triton.next_power_of_2(N_total)
    if N_total <= _MID_N:
        return 32768
    return 65536


def _launch_flat(x, out, threshold, value):
    N_total = x.numel()
    BLOCK = _choose_block(N_total)
    grid_size = min(triton.cdiv(N_total, BLOCK), NUM_SIPS * 2)
    num_warps = 4 if x.dtype == torch.float32 else 2
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared nodes);
    # retrying the launch recompiles and succeeds (rad2deg / sinh / atan2_
    # precedents for the same toolchain family).
    for attempt in range(3):
        try:
            with torch_device_fn.device(x.device):
                threshold__flat_kernel[(grid_size,)](
                    x,
                    out,
                    N_total,
                    threshold,
                    value,
                    BLOCK=BLOCK,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "threshold_ flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def threshold_(A, threshold, value):
    logger.debug("GEMS_ENFLAME THRESHOLD_")
    if (
        A.is_contiguous()
        and A.dtype in _FLOAT_DTYPES
        and _SMALL_N < A.numel() <= _INT32_SAFE_LIMIT - 65536
    ):
        _launch_flat(A, A, float(threshold), float(value))
        return A
    return threshold_func(A, threshold, value, out0=A)
