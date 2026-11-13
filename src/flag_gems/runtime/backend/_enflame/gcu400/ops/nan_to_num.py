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

NUM_SIPS = 24
INT32_MAX = torch.iinfo(torch.int32).max
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# bf16's promoted fp32 clamp chain (cvt + clamp + select) is too long to hide
# under the memory stream on very large tensors; the split2 kernel issues both
# half-tile loads before any compute and recovers ~4% there (measured on
# GCU400/L600, 512M-element bucket).  fp16/fp32 gain nothing from it.
SPLIT2_MIN_NUMEL = 1 << 28


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def nan_to_num_flat_kernel(
    x_ptr,
    out_ptr,
    nan,
    posinf,
    neginf,
    N_total,
    BLOCK: tl.constexpr,
    CLAMPABLE: tl.constexpr,
):
    # GCU400 override of the generic pointwise_dynamic path: the codegen kernel
    # lowers nan detection through tl_extra_shim.isnan (libdevice call) and a
    # three-deep tl.where chain, which measured 0.50-0.57x of torch native on
    # the 16M/1G core shapes.  When the replacement bounds lie outside the
    # dtype's finite range (the default finfo.max/min), the infinities are the
    # only values the bounds can touch, so the two where-selects collapse into
    # one clamp; NaN keeps the self-comparison test x != x.  Scalars are
    # narrowed to the element dtype so fp16 stays in native fp16 math (fp16 16M
    # bucket: 1.09x vs 0.89x with fp32 promotion).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    et = out_ptr.dtype.element_ty
    nan_t = nan.to(et)
    posinf_t = posinf.to(et)
    neginf_t = neginf.to(et)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        if CLAMPABLE:
            y = tl.clamp(x, neginf_t, posinf_t)
            y = tl.where(x != x, nan_t, y)
        else:
            y = tl.where(x != x, nan_t, x)
            y = tl.where(x == float("inf"), posinf_t, y)
            y = tl.where(x == -float("inf"), neginf_t, y)
        tl.store(out_ptr + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def nan_to_num_flat_split2_kernel(
    x_ptr, out_ptr, nan, posinf, neginf, N_total, BLOCK: tl.constexpr
):
    # Clamp-form fast path (host only dispatches here when the bounds cannot
    # clip finite values), but each iteration issues both half-tile loads
    # before any compute (manual load hoisting): on bf16 the promoted fp32
    # conversion chain otherwise serialises against the memory stream on very
    # large tensors.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    half: tl.constexpr = BLOCK // 2
    arange = tl.arange(0, half)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    et = out_ptr.dtype.element_ty
    nan_t = nan.to(et)
    posinf_t = posinf.to(et)
    neginf_t = neginf.to(et)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off0 = block_id * BLOCK + arange
        off1 = off0 + half
        m0 = off0 < N_total
        m1 = off1 < N_total
        x0 = tl.load(x_ptr + off0, mask=m0)
        x1 = tl.load(x_ptr + off1, mask=m1)
        y0 = tl.clamp(x0, neginf_t, posinf_t)
        y1 = tl.clamp(x1, neginf_t, posinf_t)
        y0 = tl.where(x0 != x0, nan_t, y0)
        y1 = tl.where(x1 != x1, nan_t, y1)
        tl.store(out_ptr + off0, y0, mask=m0)
        tl.store(out_ptr + off1, y1, mask=m1)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    x_nan = x != x
    x_posinf = x == float("inf")
    x_neginf = x == -float("inf")
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x


def _choose_block(N_total, itemsize):
    # do_bench(median) sweep, kernel mode, vs torch.nan_to_num on GCU400/L600:
    #   16M bucket peaks at 65536 (fp16 28.1us vs 28.9us @32768, bf16 34.3 vs
    #   36.5, fp32 ties); the >2^28 bucket splits by dtype -- 2-byte elements
    #   gain from 131072 (bf16 754 vs 811us @65536) while fp32 prefers 65536
    #   (1344 vs 1352us).
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if N_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _try_flat(A, nan, posinf, neginf, out=None):
    # Returns the result tensor when the flat kernel applied, else None.
    if A.dtype not in _FLAT_DTYPES or not A.is_contiguous() or A.numel() == 0:
        return None
    N_total = A.numel()
    BLOCK = _choose_block(N_total, A.element_size())
    num_blocks = triton.cdiv(N_total, BLOCK)
    if num_blocks * BLOCK > INT32_MAX:
        return None
    finfo = torch.finfo(A.dtype)
    clampable = posinf >= finfo.max and neginf <= finfo.min
    if out is None:
        out = torch.empty_like(A)
    grid_size = min(num_blocks, NUM_SIPS * 2)
    if clampable and A.dtype == torch.bfloat16 and N_total > SPLIT2_MIN_NUMEL:
        with torch_device_fn.device(A.device):
            nan_to_num_flat_split2_kernel[(grid_size,)](
                A, out, nan, posinf, neginf, N_total, BLOCK=BLOCK, num_warps=2
            )
    else:
        with torch_device_fn.device(A.device):
            nan_to_num_flat_kernel[(grid_size,)](
                A,
                out,
                nan,
                posinf,
                neginf,
                N_total,
                BLOCK=BLOCK,
                CLAMPABLE=clampable,
                num_warps=2,
            )
    return out


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_ENFLAME NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    result = _try_flat(A, nan, posinf, neginf)
    if result is not None:
        return result
    return nan_to_num_func(A, nan, posinf, neginf)


def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_ENFLAME NAN_TO_NUM_")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    result = _try_flat(A, nan, posinf, neginf, out=A)
    if result is not None:
        return result
    nan_to_num_func(A, nan, posinf, neginf, out0=A)
    return A
