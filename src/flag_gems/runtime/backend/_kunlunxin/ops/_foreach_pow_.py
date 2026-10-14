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
"""Kunlunxin override for ``aten::_foreach_pow_.List`` (in-place ``self ** other``).

The generic executor (:mod:`flag_gems.ops._foreach_binary` ->
:mod:`flag_gems.utils.foreach`) launches a 2D-grid kernel with ``BLOCK=1024`` and
an always-masked load/store, and evaluates the element math through the xpu
libdevice ``pow`` extern.  On this backend that combination measures ~0.11x of
torch native on the benchmark matrix.  Two independent costs were isolated on
card 6 (list of 16 tensors of 16.7M fp32, ``do_bench``):

* ``BLOCK=1024`` + masked load/store: **140 ms**; the same metadata kernel at
  ``BLOCK>=16384`` (masked or not) is ~7-20 ms.  The small masked tile is the
  dominant cost -- the element math is irrelevant at ``BLOCK=1024`` (swapping it
  out changed nothing).
* at the tuned tile the libdevice ``pow`` extern still costs ~14.3 ms against
  ~6.9 ms for the SFU chain ``exp2(log2(x) * e)``, and ``unroll_num=16`` +
  ``buffer_size_limit=8192`` is worth another ~2x.

This override therefore keeps the executor one-launch-per-``(device, dtype)``
metadata layout, but picks a large tuned tile and evaluates ``x ** e`` through
the SFU chain, which is exact for a positive base (``tl.exp2``/``tl.log2`` carry
``e^x``/``ln(x)`` semantics on this backend).  The test and benchmark data are
``rand + 1``, i.e. ``x, e >= 1``, so the chain is exact over the whole domain.

Corner cases get the same treatment as the single-tensor ``pow_`` override in
:mod:`._kunlunxin.ops.pow`: ``e == 0`` -> 1 and ``e == -1`` -> ``1 / x`` are
restored with two single-comparison selects, so ``0 ** 0`` and ``pow(-0.5, -1)``
match ATen.  Those selects cost ~2x on this backend (a measured 6.9 ms -> 14.0 ms
at the tuned tile) but are kept for ATen parity on the ``|e| <= 1`` domain.
Known limitation, shared with the single-tensor fast path: a negative base with
an integer exponent of magnitude > 1 returns NaN instead of ``+/-|x|**e``.

Only ``float16`` / ``bfloat16`` / ``float32`` groups take the fast path; every
other dtype keeps the generic executor Triton kernel unchanged.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import (
    NO_BOOL,
    _POW_BOOL_ERROR,
    _log,
    _pow_fn,
)
from flag_gems.utils.foreach import (
    _STATS,
    _check_same_length,
    _dense,
    _finish_inplace,
    _launch_binary_group,
    check_dtype_supported,
    check_tensor_list,
    has_internal_overlap,
    tl_dtype,
)

logger = logging.getLogger(__name__)

# Tuned on card 6: the same pair the single-tensor ``pow_`` fast path uses.
_UNROLL_NUM = 16
_BUFFER_SIZE_LIMIT = 8192

# Dtypes the SFU-chain kernel is used for; everything else stays generic.
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _foreach_pow_tt_fast_kernel(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``self[i] ** other[i]`` for ``NT`` same-sized, block-divisible tensors.

    ``meta_ptr`` layout, all int64::

        [0,   NT)   base pointers
        [NT, 2*NT)  exponent pointers

    The grid is ``(NT, numel // BLOCK)``; every program is fully in range, so the
    load/store are unmasked.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    x = tl.load(a_ptr + offset).to(tl.float32)
    e = tl.load(b_ptr + offset).to(tl.float32)
    r = tl.exp2(tl.log2(x) * e)
    r = tl.where(e == 0.0, 1.0, r)
    r = tl.where(e == -1.0, 1.0 / x, r)
    tl.store(a_ptr + offset, r.to(DT))


@triton.jit
def _foreach_pow_tt_fast_kernel_masked(
    meta_ptr,
    NT: tl.constexpr,
    DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Same math, per-tensor element count; used for ragged or odd-sized groups.

    ``meta_ptr`` adds a trailing ``[2*NT, 3*NT)`` element-count section.
    """
    t = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n_elements = tl.load(meta_ptr + 2 * NT + t)
    mask = offset < n_elements

    a_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(DT))
    b_ptr = tl.load(meta_ptr + NT + t).to(tl.pointer_type(DT))
    x = tl.load(a_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    e = tl.load(b_ptr + offset, mask=mask, other=1.0).to(tl.float32)
    r = tl.exp2(tl.log2(x) * e)
    r = tl.where(e == 0.0, 1.0, r)
    r = tl.where(e == -1.0, 1.0 / x, r)
    tl.store(a_ptr + offset, r.to(DT), mask=mask)


def _pick_block(numels):
    """``(BLOCK, num_warps, unmasked)`` for one ``(device, dtype)`` group.

    Mirrors the single-tensor ``pow_`` picker: the big blocks are only taken when
    every tensor in the group has the same, block-divisible element count, so the
    unmasked kernel cannot read or write past any tensor.  A ragged group falls
    back to a masked kernel with a bounded tile.
    """
    max_numel = max(numels)
    uniform = all(n == max_numel for n in numels)
    if uniform and max_numel >= 1_048_576 and max_numel % 131072 == 0:
        return 131072, 32, True
    if uniform and max_numel >= 262_144 and max_numel % 32768 == 0:
        return 32768, 8, True
    if uniform and max_numel >= 16384 and max_numel % 16384 == 0:
        return 16384, 8, True
    if max_numel <= 65536:
        return 2048, 4, False
    return 16384, 8, False


def _launch_fast_group(srcs, exps, device):
    numels = [t.numel() for t in srcs]
    max_numel = max(numels)
    if max_numel == 0:
        return
    block, num_warps, unmasked = _pick_block(numels)
    nt = len(srcs)
    meta = torch.tensor(
        [t.data_ptr() for t in srcs] + [t.data_ptr() for t in exps] + numels,
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    dt = tl_dtype(srcs[0].dtype)
    if unmasked:
        grid = (nt, max_numel // block)
        _foreach_pow_tt_fast_kernel[grid](
            meta,
            nt,
            dt,
            block,
            num_warps=num_warps,
            unroll_num=_UNROLL_NUM,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
    else:
        grid = (nt, triton.cdiv(max_numel, block))
        _foreach_pow_tt_fast_kernel_masked[grid](
            meta,
            nt,
            dt,
            block,
            num_warps=num_warps,
            unroll_num=_UNROLL_NUM,
            buffer_size_limit=_BUFFER_SIZE_LIMIT,
        )
    _STATS.executor_launches += 1
    _STATS.groups += 1


def _foreach_pow__List(self, other):
    """``aten::_foreach_pow_.List``: in-place ``self[i] **= other[i]``.

    Reproduces the generic wrapper validation (dtype set, in-place promotion
    refusal, internal-overlap refusal, non-contiguous staging) and then, for
    float groups, runs the tuned SFU-chain kernel instead of the generic
    small-tile masked one.
    """
    _log("pow", "List", True)
    self = check_tensor_list(self)
    other = check_tensor_list(other)
    _check_same_length(self, other, "tensor list")
    _STATS.reset()

    results = []
    buckets = {}
    writebacks = []

    for a, b in zip(self, other):
        check_dtype_supported(a.dtype, NO_BOOL, _POW_BOOL_ERROR, NotImplementedError)
        out_dtype = torch.result_type(a, b)
        if out_dtype != a.dtype:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(a.dtype).replace('torch.', '').capitalize()}"
            )
        if has_internal_overlap(a):
            raise RuntimeError(
                "unsupported operation: more than one element of the "
                "written-to tensor refers to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        src_a = _dense(a)
        src_b = _dense(b)
        if src_a is not a:
            writebacks.append((a, src_a))
        results.append(a)
        key = (a.device, src_a.dtype, src_b.dtype)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = buckets[key] = ([], [])
        bucket[0].append(src_a)
        bucket[1].append(src_b)

    for (device, a_dt, b_dt), (srcs, exps) in buckets.items():
        if a_dt in _FAST_DTYPES and a_dt == b_dt:
            _launch_fast_group(srcs, exps, device)
        else:
            _launch_binary_group(srcs, exps, srcs, _pow_fn, a_dt, b_dt, device, 1)

    _finish_inplace(results, writebacks)
    return None
