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

"""Element-wise greater-or-equal, GCU400-optimized.

The previous delegation to the public ``ge`` implementation ran through
``pointwise_dynamic`` whose ENFLAME ``CodeGenConfig`` caps the launch grid at
``(12, 1, 1)``; every shape beyond ~2**24 elements was serialized onto 12 CTAs
and reached only ~240-425 GB/s versus ~1.4-2.8 TB/s for the native kernel
(bench core, L600: greater_equal 0.215x / greater_equal_scalar 0.148x /
greater_equal_ 0.489x).

This file keeps the generic pointwise path for small / non-contiguous /
broadcast / non-float inputs, and dispatches large contiguous same-dtype float
inputs to dedicated flat 1D kernels (L600 probe, kernel mode do_bench):

* tensor & in-place share one kernel; out-of-place writes bool, in-place casts
  the result back into ``A`` (same semantics as ``ge_func(..., out0=A)``).
* comparisons run in the native input dtype (exact for same-dtype floats;
  the fp32 up-cast of the public kernel only wastes ALU here), and the
  scalar overload casts the Python scalar to ``A.dtype`` before comparing,
  which matches torch TensorIterator semantics; the public fp32-compare
  kernel deviates from torch on fp16 round-down scalars (e.g. 0.1).
* per-dtype DSM cap (max_dsm = 917504 B, compile-time enforced) bounds the
  tile size; a coverage/chunk loop kernel extends per-CTA coverage beyond
  the single-tile DSM limit for the largest shapes.

Probe-validated tiers (latency vs torch native, do_bench median):
  tensor/inplace n < 2**16   -> generic pointwise (launch-overhead floor)
  tensor/inplace 2**16..2**22 -> flat BLOCK=16384, nw=4   (~1.4-2.1x)
  tensor/inplace 2**22..2**28 (f16) -> flat BLOCK=131072, nw=4 (~0.96-1.12x)
  tensor/inplace n >= 2**22 (f32)   -> loop COV=262144 CHUNK=65536 nw=8 (~0.96-0.98x)
  tensor/inplace n >= 2**28 (f16/bf16) -> loop COV=524288 CHUNK=131072 nw=4 st=2 (~0.94-0.98x)
  scalar 0 < n < 2**22       -> flat BLOCK=16384, nw=4
  scalar n >= 2**22          -> flat BLOCK=262144 (f16/bf16) / 131072 (f32) (~0.98-1.06x)
``ge_func`` / ``ge_func_scalar`` (public pointwise kernels) remain the
fallback, so untuned corners keep their previous behaviour.
"""  # b2

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.ge import ge_func, ge_func_scalar
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 16
_GRID_CAP = NUM_SIPS * 3
_TINY_N = 4096
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
_INT32_SAFE_LIMIT = 0x7FFFFFFF


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def geq_flat_kernel(
    A, B, OUT, N_total, BLOCK: tl.constexpr, EXACT: tl.constexpr, NS: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(A + off)
            y = tl.load(B + off)
            r = x >= y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(A + off, mask=mask, other=0)
            y = tl.load(B + off, mask=mask, other=0)
            r = x >= y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def geq_scalar_flat_kernel(
    A, OUT, N_total, Y, BLOCK: tl.constexpr, EXACT: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(A + off)
            r = x >= Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(A + off, mask=mask, other=0)
            r = x >= Y.to(x.dtype)
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def geq_scalar_inplace_flat_kernel(A, OUT, N_total, Y, BLOCK: tl.constexpr):
    """Scalar compare written back into ``A`` with a mask only on the tail.

    ``geq_scalar_flat_kernel`` masks every lane of every block whenever
    ``N_total`` is not a multiple of ``BLOCK``, which costs ~25% of the streaming
    bandwidth on GCU400 (fp16 5.9M: 25.6us masked vs 20.2us unmasked).  Here the
    full blocks run unmasked and only the single trailing partial block is
    masked, executed by CTA 0.  Neither scheme lets two CTAs write into the same
    region: an overlapping tail was measured to corrupt 1-byte stores, where a
    narrow store is a read-modify-write of a wider hardware granule.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_full = N_total // BLOCK
    for block_id in tl.range(pid, num_full, num_pids):
        off = block_id * BLOCK + arange
        x = tl.load(A + off)
        r = x >= Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))
    tail_start = num_full * BLOCK
    if pid == 0 and tail_start < N_total:
        off = tail_start + arange
        mask = off < N_total
        x = tl.load(A + off, mask=mask, other=0)
        r = x >= Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=mask)


def _geq_scalar_inplace_config(n, itemsize):
    """Tile shape for the in-place scalar comparison.

    The in-place overload streams one tensor in and writes the same dtype back
    (2x the store traffic of the bool-out scalar kernel), and a 1-byte dtype
    needs many more lanes than an fp32 stream to reach the same bandwidth.
    Probe results per element size (median, GB/s normalised):
      1B:  n > 2**22 -> 32768 lanes, nw4   (32K/nw2 loses 24%)
      2B:  n > 2**21 -> 131072 lanes, nw4  (64K loses 10%)
      4B:  n > 2**21 -> 32768 lanes, nw2
    num_stages does not move any of these arms.
    """
    if n <= _TINY_N:
        return max(triton.next_power_of_2(n), 256), (1 if itemsize == 2 else 2)
    if itemsize == 1:
        if n <= (1 << 22):
            return 16384, 4
        return 32768, 4
    if itemsize == 2:
        if n <= (1 << 17):
            return 8192, 4
        if n <= (1 << 21):
            return 65536, 4
        return 131072, 4
    if n <= (1 << 21):
        return 16384, 4
    return 32768, 2


def _geq_pair_config(n, itemsize, inplace=False):
    if n <= _TINY_N:
        # fp32 needs nw2 here: one warp holding 128 fp32 lanes per operand
        # serializes (4096/nw1 = 6.4us vs 3.77us at nw2); 16-bit is fine at nw1.
        return max(triton.next_power_of_2(n), 256), (1 if itemsize == 2 else 2), 1
    if itemsize == 2:
        if n <= (1 << 17):
            return 8192, 4, 1
        if n <= (1 << 23):
            return 16384, 4, 1
        # ns=1 beats ns=3 on the flat 1Gi case in-session (ns3 won by ~1% on
        # the gt twin's card; the effect is card-state dependent and ns1 never
        # loses the paired probe here: f16-flat -1.9%, bf16-flat -3.4%).
        return 65536, 2, 1
    if n <= (1 << 21):
        return 8192, 4, 1
    if n <= (1 << 27):
        return 32768, 2, 1
    if inplace:
        # The bool-out 16384/nw1 arm collapses on the 12B/elem dtype-out
        # stream (6160 vs 4223us paired probe, 2026-11-16): a single warp
        # cannot keep the 2x fp32 load + fp32 store footprint in flight.
        # In-session sweep on gcu400 (1 Gi fp32 in-place): 65536/nw4 4.80 ms,
        # 65536/nw2 4.86 ms, 32768/nw4 4.94 ms, 32768/nw2 4.98 ms.
        return 65536, 4, 1
    return 16384, 1, 1


def _geq_scalar_config(n, itemsize):
    if n <= _TINY_N:
        # Same fp32 nw1 serialization as the pair kernel (5.3us -> 3.84us at nw2).
        return max(triton.next_power_of_2(n), 256), (1 if itemsize == 2 else 2)
    if n <= (1 << 22):
        return 16384, 4
    if itemsize == 2:
        return 65536, 2
    if n <= (1 << 25):
        return 32768, 2
    return 65536, 1


def _geq_launch(kernel, grid, args, block, num_warps, **constexprs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (add.py / greater.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kernel[(grid,)](
                    *args,
                    BLOCK=block,
                    num_warps=num_warps,
                    num_stages=1,
                    **constexprs,
                )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning(
                "greater_equal flat kernel launch failed (attempt %d)",
                attempt + 1,
                exc_info=True,
            )


def _geq_pair_run(A, B, out, inplace=False):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n == 0:
        return False
    block, num_warps, num_stages = _geq_pair_config(n, A.element_size(), inplace)
    if n > _INT32_SAFE_LIMIT - block:
        return False
    grid = min(-(-n // block), _GRID_CAP)
    _geq_launch(
        geq_flat_kernel,
        grid,
        (A, B, out, n),
        block,
        num_warps,
        EXACT=n % block == 0,
        NS=num_stages,
    )
    return True


def greater_equal(A, B):
    logger.debug("GEMS_ENFLAME GREATER_EQUAL")
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_like(A, dtype=torch.bool)
        if _geq_pair_run(A, B, out):
            return out
    return ge_func(A, B)


def greater_equal_scalar(A, B):
    logger.debug("GEMS_ENFLAME GREATER_EQUAL_SCALAR ")  # b2
    if isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous():
        n = A.numel()
        if n > 0:
            block, num_warps = _geq_scalar_config(n, A.element_size())
            if n <= _INT32_SAFE_LIMIT - block:
                out = torch.empty_like(A, dtype=torch.bool)
                grid = min(-(-n // block), _GRID_CAP)
                _geq_launch(
                    geq_scalar_flat_kernel,
                    grid,
                    (A, out, n, float(B)),
                    block,
                    num_warps,
                    EXACT=n % block == 0,
                )
                return out
    return ge_func_scalar(A, B)


def greater_equal_(A, B):
    logger.debug("GEMS_ENFLAME GREATER_EQUAL_")
    if not isinstance(B, torch.Tensor):
        # aten dispatches ``Tensor.greater_equal_(scalar)`` onto the Tensor
        # overload, so B can arrive as a Python number. Compare through the
        # scalar pointwise kernel and write the result back into A; the kernel
        # stores the bool result into A.dtype, which matches aten for byte and
        # floating tensors alike.
        return ge_func_scalar(A, B, out0=A)
    if A.device != B.device:
        B = B.to(A.device)
    if (
        A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        if _geq_pair_run(A, B, A, inplace=True):
            return A
    return ge_func(A, B, out0=A)


def greater_equal_tensor_(A, B):
    """``aten::greater_equal_.Tensor`` entry point.

    ``flag_gems`` binds this overload to the public ``greater_equal_tensor_``,
    which always routes through the generic ``ge_func`` pointwise kernel and
    therefore never reaches the flat GCU400 kernel above (measured 13.4 ms vs
    4.6 ms for the 1 Gi fp32 in-place case).  Exposing the name here makes the
    vendor registrar bind the Tensor overload to the tuned implementation.
    """
    return greater_equal_(A, B)


def greater_equal_scalar_(A, B):
    """``aten::greater_equal_.Scalar``: compare against a Python number in place.

    The result is written back into ``A`` with ``A``'s own dtype (1/0), which is
    what ``ge_func_scalar(..., out0=A)`` produces.  Contiguous float and integer
    inputs use the flat scalar kernel; everything else (broadcast, non
    contiguous, bool, values outside the native dtype range) keeps the generic
    pointwise path.
    """
    logger.debug("GEMS_ENFLAME GREATER_EQUAL_SCALAR_")
    if isinstance(B, (int, float)) and not isinstance(B, bool) and A.is_contiguous():
        dtype = A.dtype
        supported = dtype in _FLOAT_DTYPES or dtype in _INT_DTYPES
        value = float(B)
        if supported and dtype in _INT_DTYPES:
            info = torch.iinfo(dtype)
            # A scalar that does not fit the native dtype would be saturated by
            # the kernel cast, so keep the wider generic comparison there.
            supported = info.min <= value <= info.max and value == int(value)
        if supported:
            n = A.numel()
            if n > 0:
                block, num_warps = _geq_scalar_inplace_config(n, A.element_size())
                if n <= _INT32_SAFE_LIMIT - block:
                    grid = min(-(-n // block), _GRID_CAP)
                    if block <= n:
                        _geq_launch(
                            geq_scalar_inplace_flat_kernel,
                            grid,
                            (A, A, n, value),
                            block,
                            num_warps,
                        )
                        return A
    return ge_func_scalar(A, B, out0=A)
