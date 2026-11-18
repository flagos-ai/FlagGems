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

"""Element-wise equality, GCU400-optimized (eq / eq_scalar; `equal` kept as-is).

The previous path ran every shape through the pointwise_dynamic codegen
template (bptr tiles, fixed num_warps=4) whose fp32 up-cast comparison wastes
ALU on same-dtype floats and whose template store structure leaves ~20%
bandwidth on the table versus native (L600 QA report: eq 0.768x /
eq_scalar 0.745x; mid-band 2^24 shapes re-measured at 0.74-0.92x).

Large contiguous same-dtype float inputs are dispatched to dedicated flat 1D
kernels instead (probe-validated tiers, sibling of greater_equal/copysign):

* comparisons run in the native input dtype -- bit-exact for same-dtype
  floats (the fp32 up-cast is an injective map, so `==` is preserved) and
  cheaper (the up-cast costs 17-22% on f16/bf16);
* the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator semantics; the fp32-compare
  template deviates from torch on fp16 round-down scalars (e.g. 0.1);
* per-dtype DSM cap (flat dual-load Required = 2 x itemsize x BLOCK,
  917504 B compile-time limit) bounds BLOCK; a coverage/chunk loop kernel
  extends per-CTA coverage past the single-tile DSM cap for giant shapes.

Probe-validated tiers (L600, do_bench kernel-mode median vs torch native;
three interleaved sweep rounds, 2026-11-11):

  tensor   n < 2^16            -> generic pointwise (launch-overhead floor)
  tensor   2^16..2^22          -> flat BLOCK=65536 nw4            (~1.3-1.6x)
  tensor   2^22..2^28 (f16)    -> flat BLOCK=131072 nw4           (~0.96-1.0x)
  tensor   2^22..2^28 (f32)    -> loop COV=262144 CHUNK=65536 nw8 (~0.95x)
  tensor   n >= 2^28 (f16)      -> loop COV=1048576 CHUNK=131072 nw2 st2 (~1.0x)
  tensor   n >= 2^28 (f32)      -> strided BLOCK=65536 grid=48 nw4  (~1.0x)
  scalar   n < 2^16            -> flat BLOCK=next_pow2(n) nw4      (~1.5x)
  scalar   2^16..2^22          -> flat BLOCK=65536 nw4            (~1.2-1.3x)
  scalar   n >= 2^22 (f16/bf16) -> flat BLOCK=262144 nw4           (~1.0-1.06x)
  scalar   2^22..2^28 (f32)    -> flat BLOCK=131072 nw4           (~1.01x)
  scalar   n >= 2^28 (f32)      -> strided BLOCK=65536 grid=48 nw4  (~0.98-0.99x)

The fp32 giant arms use a strided (grid-stride, 48 persistent CTAs) tile
assignment: under the official bench's sustained traffic the contiguous
coverage-chunk loop loses ~5% there, while f16 giants prefer the chunk loop.
"""

import logging
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

_NP2 = triton.next_power_of_2
_NPROGS = 48
_BS = 32768

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SMALL_N = 2**16
_MID_N = 2**22
_BIG_N = 2**28
_INT32_SAFE_LIMIT = 0x7FFFFFFF


def _launch_with_retry(launch):
    # Flaky external toolchain (fresh-compile segv windows on the shared
    # node): retry compile-time failures only, re-raise the original error
    # once exhausted so real failures are never masked.
    for attempt in range(6):
        try:
            launch()
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(1.0)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


@triton.jit
def _eq_cmp_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    y = tl.load(B + off, mask=m, other=0)
    r = x == y
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _eq_cmp_loop_kernel(
    A, B, OUT, N, COV: tl.constexpr, CHUNK: tl.constexpr, ST: tl.constexpr
):
    base = tl.program_id(0) * COV
    for j in tl.range(0, COV // CHUNK, num_stages=ST):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x == y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _eq_cmp_strided_kernel(A, B, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for tile in tl.range(pid, tl.cdiv(N, BLOCK), num_ctas):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        y = tl.load(B + off, mask=m, other=0)
        r = x == y
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _eq_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(A + off, mask=m, other=0)
    r = x == Y.to(x.dtype)
    tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _eq_cmp_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n < _SMALL_N:
        return False
    if A.dtype == torch.float32:
        if n >= _BIG_N:
            # strided tile assignment (48 persistent CTAs): under sustained
            # traffic the contiguous-chunk loop loses ~5% on fp32 giant shapes
            if n > _INT32_SAFE_LIMIT - 65536:
                return False
            _launch_with_retry(
                lambda: _eq_cmp_strided_kernel[(_NPROGS,)](
                    A, B, out, n, BLOCK=65536, num_warps=4
                )
            )
            return True
        if n >= _MID_N:
            cov, chunk, nw, st = 262144, 65536, 8, 1
        else:
            cov = None
        if cov is not None:
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _launch_with_retry(
                lambda: _eq_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                    A,
                    B,
                    out,
                    n,
                    COV=cov,
                    CHUNK=chunk,
                    ST=st,
                    num_warps=nw,
                    num_stages=st,
                )
            )
            return True
        block, nw = 65536, 4
    else:
        if n >= _BIG_N:
            cov, chunk, nw, st = 1048576, 131072, 2, 2
            if n > _INT32_SAFE_LIMIT - cov:
                return False
            _launch_with_retry(
                lambda: _eq_cmp_loop_kernel[(triton.cdiv(n, cov),)](
                    A,
                    B,
                    out,
                    n,
                    COV=cov,
                    CHUNK=chunk,
                    ST=st,
                    num_warps=nw,
                    num_stages=st,
                )
            )
            return True
        block, nw = (131072, 4) if n >= _MID_N else (65536, 4)
    if n > _INT32_SAFE_LIMIT - block:
        return False
    _launch_with_retry(
        lambda: _eq_cmp_kernel[(triton.cdiv(n, block),)](
            A, B, out, n, BLOCK=block, num_warps=nw
        )
    )
    return True


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS_ENFLAME EQ")
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _eq_cmp_run(A, B, out):
            return out
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar(x, y):
    return x.to(tl.float32) == y.to(tl.float32)


def eq_scalar(A, B):
    logger.debug("GEMS_ENFLAME EQ_SCALAR")
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        try:
            y = float(B)
        except (OverflowError, ValueError):
            return eq_func_scalar(A, B)
        n = A.numel()
        if n > 0:
            if n >= _MID_N:
                if A.dtype == torch.float32 and n >= _BIG_N:
                    if n > _INT32_SAFE_LIMIT - 65536:
                        return eq_func_scalar(A, B)
                    out = torch.empty_like(A, dtype=torch.bool)
                    _launch_with_retry(
                        lambda: _eq_cmp_scalar_strided_kernel[(_NPROGS,)](
                            A, out, n, y, BLOCK=65536, num_warps=4
                        )
                    )
                    return out
                block = 262144 if A.dtype != torch.float32 else 131072
            elif n >= _SMALL_N:
                block = 65536
            else:
                # shape-matched tile for launch-floor shapes: a fixed 16384-wide
                # vector wastes 4x lanes on (64,64)-class fp32 inputs (+1.5us)
                block = max(256, min(16384, _NP2(n)))
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                _launch_with_retry(
                    lambda: _eq_cmp_scalar_kernel[(triton.cdiv(n, block),)](
                        A, out, n, y, BLOCK=block, num_warps=4
                    )
                )
                return out
    return eq_func_scalar(A, B)


@triton.jit
def _eq_cmp_scalar_strided_kernel(A, OUT, N, Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    for tile in tl.range(pid, tl.cdiv(N, BLOCK), num_ctas):
        off = tile * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        r = x == Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _equal_small_k(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    all_eq = tl.full((), 1, dtype=tl.int32)
    for off in range(0, numel, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        eq_val = tl.min((x == y).to(tl.int32), axis=0)
        all_eq = all_eq & eq_val
    tl.store(out_ptr, all_eq)


@triton.jit
def _equal_fused_k(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    num_progs,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    all_eq = tl.full((), 1, dtype=tl.int32)
    blk = pid
    while blk * BLOCK_SIZE < numel:
        offsets = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
        eq_val = tl.min((x == y).to(tl.int32), axis=0)
        all_eq = all_eq & eq_val
        blk += num_progs
    tl.store(out_ptr + pid, all_eq)


@triton.jit
def _reduce_min_k(in_ptr, out_ptr, N: tl.constexpr):
    idx = tl.arange(0, N)
    v = tl.load(in_ptr + idx)
    result = tl.min(v, axis=0)
    tl.store(out_ptr, result)


def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    logger.debug("GEMS_ENFLAME EQUAL")
    if x.shape != y.shape:
        return False
    if x.numel() == 0:
        return True

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    numel = x_flat.numel()

    out = torch.empty(1, dtype=torch.int32, device=x.device)

    if numel <= _BS:
        _equal_small_k[(1,)](
            x_flat,
            y_flat,
            out,
            numel,
            BLOCK_SIZE=_NP2(numel),
            num_warps=1,
        )
    else:
        NP = _NP2(_NPROGS)
        out_t = torch.ones(NP, dtype=torch.int32, device=x.device)
        _equal_fused_k[(_NPROGS,)](
            x_flat,
            y_flat,
            out_t,
            numel,
            _NPROGS,
            BLOCK_SIZE=_BS,
            num_warps=1,
        )
        _reduce_min_k[(1,)](out_t, out, N=NP, num_warps=1)

    return bool(out.item() > 0)
