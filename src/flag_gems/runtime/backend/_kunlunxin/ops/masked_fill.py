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
import math

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import broadcastable_to

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic pointwise path (fallback for non-contiguous inputs, broadcastable
# masks, tensor values and tiny shapes). Tuned on XPU: isCloseVectorization
# keeps the mixed i1-mask/dtype tl.where vectorized.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


# Small/mid flat shapes (single-CTA trap region): the generated autoGrid path
# launches ONE CTA for N<=2048*64=131072 (pointwise_dynamic.py:884-896), which
# serializes the whole fill on a single cluster. A hand-written constexpr
# monolithic kernel at grid=12 (= P800 cluster_num, full machine) with
# BLOCK=next_power_of_2(cdiv(N, 12)) measures 16-55% faster than the generated
# single-CTA launch in 8192<=N<=131072 (probe10, do_bench median):
#   N=8192: 6.59/6.15us vs wrapper 6.90/6.76 (fp16/fp32)
#   N=10000: 6.76/6.54 vs 8.04/7.75
#   N=65536: 8.43/8.32 vs 15.45/14.87
#   N=131072: 11.22/10.69 vs 25.20/23.19
# Below 8192 the single-CTA generated kernel is already at the launch floor
# (N=4096: mono 7.07 vs wrapper 6.34) and above 131072 the generated 12-CTA
# path's block-DMA pipelining wins decisively (N>=262144: mono 1.0-2.8x slower),
# so this kernel is used strictly inside the trap region.
@triton.jit(do_not_specialize=["value"])
def masked_fill_small_kernel(
    inp_ptr, mask_ptr, value, out_ptr, num_tasks: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < num_tasks
    m = tl.load(mask_ptr + offs, mask=in_bounds, other=0)
    x = tl.load(inp_ptr + offs, mask=in_bounds, other=0.0)
    r = tl.where(m != 0, value, x)
    tl.store(out_ptr + offs, r, mask=in_bounds)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, "NO_OPMATH")],
    config=_config,
)
@triton.jit
def masked_fill_tensor_value_kernel(inp, expand_mask, value):
    return tl.where(expand_mask, value, inp)


# ---------------------------------------------------------------------------
# Flat fast path (scalar value, contiguous fp16/bf16/fp32, numel >= gate).
#
# One 1:1 pass over x / mask bytes / out with a per-lane select in the tensor's
# own dtype, bit-identical to `where(mask, V, x)` for every input - including
# masked lanes whose x is NaN/Inf, which ATen overwrites with V.
#
# An arithmetic blend,
#   r = x * (1 - m) + V * m               (m = mask byte 0/1 as float)
# is ~1.5-2x faster on XPU (16.8M fp16 elems: 57us vs 122us) because it avoids
# the per-lane `sel`, but `0 * inf = nan` makes masked non-finite lanes come out
# NaN instead of V, so it cannot be used without changing masked_fill semantics.
_FAST_TILE = 131072
_FAST_MIN_NUMEL = 1 << 20
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def masked_fill_fast_kernel(
    out_ptr, x_ptr, mask_ptr, V: tl.constexpr, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    x = tl.load(x_ptr + tid)
    m = tl.load(mask_ptr + tid)
    vv = tl.full((), V, x.dtype)
    tl.store(out_ptr + tid, tl.where(m != 0, vv, x))


@triton.jit
def masked_fill_fast_masked_kernel(
    out_ptr, x_ptr, mask_ptr, numel, V: tl.constexpr, TILE: tl.constexpr
):
    pid = tl.program_id(0)
    tid = pid * TILE + tl.arange(0, TILE)
    m0 = tid < numel
    x = tl.load(x_ptr + tid, mask=m0)
    m = tl.load(mask_ptr + tid, mask=m0)
    vv = tl.full((), V, x.dtype)
    tl.store(out_ptr + tid, tl.where(m != 0, vv, x), mask=m0)


def _masked_fill_fast(inp, mask, value, out):
    n = inp.numel()
    # Round the fill value to the tensor dtype first (torch does the same cast),
    # so the splatted constant is bit-identical to ATen's fill value.
    v = float(torch.tensor(value, dtype=inp.dtype).item())
    mask8 = mask.view(torch.int8)
    launch = dict(
        num_warps=4,
        buffer_size_limit=8192,
        unroll_num=16,
        isCloseMemoryAsync=False,
    )
    if n % _FAST_TILE == 0:
        masked_fill_fast_kernel[(n // _FAST_TILE,)](
            out, inp, mask8, V=v, TILE=_FAST_TILE, **launch
        )
    else:
        masked_fill_fast_masked_kernel[(math.ceil(n / _FAST_TILE),)](
            out, inp, mask8, n, V=v, TILE=_FAST_TILE, **launch
        )
    return out


def _use_fast_path(inp, mask, value):
    if torch.is_tensor(value):
        return False
    if inp.dtype not in _FAST_DTYPES:
        return False
    if not (inp.is_contiguous() and mask.is_contiguous()):
        return False
    if tuple(mask.shape) != tuple(inp.shape):
        return False
    return inp.numel() >= _FAST_MIN_NUMEL


def _normalize_bool_scalar(value):
    """Hoist a Python ``bool`` value to ``int`` before it reaches any kernel.

    ``value`` is carried into the Triton kernels as an untyped scalar under
    ``do_not_specialize`` (both the generated ``pointwise_dynamic`` kernels and
    ``masked_fill_small_kernel``).  Its element type is bound once per process,
    on the first call: a leading Python ``bool`` binds it to ``i1`` and every
    later integer scalar is then silently truncated with ``& 1``
    (``masked_fill_(mask, True)`` followed by ``masked_fill_(mask, 3)`` fills
    ``1.0`` instead of ``3.0``; ``-7`` and ``2`` give ``1.0`` and ``0.0``).
    ``bool`` is exactly representable as ``int`` 0/1, so hoisting it removes the
    poisoned binding without changing any result.
    """
    return int(value) if isinstance(value, bool) else value


def masked_fill(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL")
    value = _normalize_bool_scalar(value)
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        if value.device != inp.device:
            raise RuntimeError("masked_fill value must be on the input device")
        kernel = masked_fill_tensor_value_kernel
    else:
        kernel = masked_fill_kernel
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        out = torch.empty_like(inp)
        kernel(inp, mask, value, out0=out)
        return out

    out = torch.empty_like(inp, dtype=inp.dtype, device=inp.device)
    if inp.numel() == 0:
        return out

    if _use_fast_path(inp, mask, value):
        return _masked_fill_fast(inp, mask, value, out)

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        # Common case (mask matches inp): one flat stride-1 pass, which is
        # what the tuned 1D config accelerates.
        mask = mask.contiguous()
        flat_in = inp.view(-1)
        flat_mask = mask.view(-1)
        flat_out = out.view(-1)
        num_tasks = flat_in.numel()
        if kernel is masked_fill_kernel and 8192 <= num_tasks <= 131072:
            # Single-CTA trap region: use the full-machine monolithic kernel.
            block = triton.next_power_of_2(triton.cdiv(num_tasks, 12))
            masked_fill_small_kernel[(12,)](
                flat_in,
                flat_mask,
                value,
                flat_out,
                num_tasks,
                BLOCK=block,
                num_warps=4,
                buffer_size_limit=4096,
                isCloseVectorization=True,
            )
        else:
            kernel(flat_in, flat_mask, value, out0=flat_out)
    else:
        expand_mask = mask.expand(inp.shape)
        kernel.instantiate(inp.ndim)
        kernel(inp, expand_mask, value, out0=out)
    return out


def masked_fill_(inp, mask, value):
    logger.debug("GEMS_KUNLUNXIN MASKED_FILL_")
    value = _normalize_bool_scalar(value)
    assert (
        (torch.is_tensor(value) and value.ndim == 0)
        or isinstance(value, int)
        or isinstance(value, float)
    ), "masked_fill_ only supports a 0-dimensional value tensor"
    if torch.is_tensor(value):
        if value.device != inp.device:
            raise RuntimeError("masked_fill value must be on the input device")
        kernel = masked_fill_tensor_value_kernel
    else:
        kernel = masked_fill_kernel
    assert broadcastable_to(
        mask.shape, inp.shape
    ), "The shape of mask must be broadcastable with the shape of the underlying tensor"

    if inp.ndim == 0:
        kernel(inp, mask, value, out0=inp)
        return inp

    if inp.numel() == 0:
        return inp

    if _use_fast_path(inp, mask, value):
        return _masked_fill_fast(inp, mask, value, inp)

    if inp.is_contiguous() and tuple(mask.shape) == tuple(inp.shape):
        mask = mask.contiguous()
        flat_in = inp.view(-1)
        flat_mask = mask.view(-1)
        num_tasks = flat_in.numel()
        if kernel is masked_fill_kernel and 8192 <= num_tasks <= 131072:
            # Same single-CTA trap-region rule as out-of-place masked_fill: the
            # generated autoGrid path would launch ONE CTA here, while the
            # monolithic kernel fills the machine with 12 CTAs. In-place is
            # safe: every program loads its disjoint tile before storing it,
            # and out_ptr aliases flat_in element-for-element.
            block = triton.next_power_of_2(triton.cdiv(num_tasks, 12))
            masked_fill_small_kernel[(12,)](
                flat_in,
                flat_mask,
                value,
                flat_in,
                num_tasks,
                BLOCK=block,
                num_warps=4,
                buffer_size_limit=4096,
                isCloseVectorization=True,
            )
        else:
            kernel(flat_in, flat_mask, value, out0=flat_in)
    else:
        expand_mask = mask.expand(inp.shape)
        kernel.instantiate(inp.ndim)
        kernel(inp, expand_mask, value, out0=inp)
    return inp
