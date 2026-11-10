# Copyright 2026, The FlagOS Contributors.
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
"""Kunlunxin (XPU) L1 loss.

The generic implementation folds the per-block partials with a single-program
``tl.sum`` whose width is ``next_power_of_2(numel / 1024)``.  On this backend a
*masked* 1-D reduce is silently miscompiled once the tile width reaches 16384
lanes (e.g. ``next_pow2(24023) == 32768`` returns ~0 instead of 2.7e7), which
corrupts the ``mean``/``sum`` reductions for large tensors.

This override keeps every reduce tile at ``_BLOCK = 1024`` lanes (verified
correct for both masked and unmasked loads), pads the per-block partials so the
folding stages never need a masked reduce, and folds in bounded stages.

Range-reduction trick for accumulation accuracy
-----------------------------------------------
Summing ~2.5e7 fp32 values of magnitude ~1 in a plain fp32 accumulate loses the
low bits (measured ~1.3e3 absolute error vs. the correctly-rounded result,
which fails the 1.3e-6 rtol in the accuracy suite).  Because the L1 residual
``|x - y|`` for typical inputs is O(1), the kernel subtracts 1.0 from every
residual before summing and adds ``n`` back at the end:

    sum |x - y| = sum (|x - y| - 1) + n

The residual sums are two to three orders of magnitude smaller than the raw
total, so the same fp32 tree reduce lands within ~1e1 of the exact value (which
is what the reference produces).  The identity is exact in real arithmetic, so
this is a pure range reduction, not an approximation.

Wide-tile vector accumulator
----------------------------
Program count dominates this backend: 1024-lane tiles over 16k programs run
~6x slower than a few hundred programs for the same element count.  Each
program therefore owns up to 8 chunks of 16384 lanes (128 programs for a 16.7M
element tensor).  The partial sums are kept as a *vector* of 16384 fp32 lanes
that each lane accumulates across its chunks, and the tree reduce runs once
per program; reducing every chunk separately costs ~2.7x (fp16 16.7M: 220us
vs 80us).  The ``|x - y|`` residual is formed in the *input* dtype and
promoted to fp32 once per chunk: for fp16/bf16 inputs this lifts the kernel
from ~310 GB/s to ~640-850 GB/s (the reference computes fp32, so the fp16
rounding is far inside the 1e-3 / 1.6e-2 rtol of those dtypes).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# Safe 1-D reduce width on this backend (masked loads miscompile at
# >= 16384 lanes; 1024 is also the width the platform's online-reduce
# kernels are known to compile reliably at).
_BLOCK = 1024
# Width used for the folding stages; equal to _BLOCK so the same compiled
# tiles are reused and no masked reduce is ever issued.
_FOLD = 1024
# Fast path: one program owns ITER unmasked chunks of BLOCK lanes.  The
# accumulator is a *vector* of BLOCK fp32 lanes that is tree-reduced once at
# the end of the program; reducing every chunk separately costs ~2.7x on
# this backend (fp16 16.7M: 220us vs 80us).
_FAST_BLOCKS = (16384, 8192, 4096, 2048, 1024)
_FAST_ITER_MAX = 8
_FAST_WARPS = 8
# Size of the single-program path (whole reduction in one launch).
_SMALL_MAX = 8192


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _l1_loss_elementwise_kernel(x, y):
    return tl.abs(x.to(tl.float32) - y.to(tl.float32))


@triton.jit
def _l1_loss_partial_sum_kernel(inp, target, mid, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    inp_val = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target + offsets, mask=mask, other=0.0).to(tl.float32)
    # Range reduction: accumulate (|d| - 1); ``n`` is added back on the host.
    diff = tl.abs(inp_val - target_val) - 1.0
    diff = tl.where(mask, diff, 0.0)
    acc = tl.sum(diff, axis=0)
    tl.store(mid + pid, acc)


@triton.jit
def _l1_loss_fold_kernel(mid, out, BLOCK: tl.constexpr):
    """Unmasked fold of a padded partial buffer; writes one value per program."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(mid + offsets).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(out + pid, acc)


@triton.jit
def _l1_loss_part_kernel(
    inp,
    target,
    buf,
    denom,
    DIV: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK: tl.constexpr,
    ITER: tl.constexpr,
):
    """Unmasked wide tile: ``ITER`` sequential chunks of ``BLOCK`` lanes.

    Every lane is valid by construction (the host only takes this path when
    ``n % BLOCK == 0``), so no predicate is emitted and the reduce width
    stays at the safe ``BLOCK`` lanes.  Each lane accumulates its own column
    across the chunks and the single tree reduce happens once per program;
    the residual is formed in the input dtype and promoted once per chunk.
    """
    pid = tl.program_id(axis=0)
    base = pid * (ITER * BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(ITER):
        offsets = base + i * BLOCK + tl.arange(0, BLOCK)
        inp_val = tl.load(inp + offsets)
        target_val = tl.load(target + offsets)
        # Range reduction: (|d| - 1) is summed; ``n`` is added back on the host.
        diff = tl.abs(inp_val - target_val).to(tl.float32)
        acc += diff - 1.0
    total = tl.sum(acc, axis=0)
    if FINAL:
        if DIV:
            total = total / denom + 1.0
        else:
            total = total + denom
        tl.store(buf, total)
    else:
        tl.store(buf + pid, total)


@triton.jit
def _l1_loss_final_kernel(
    mid, out, denom, n_parts, DIV: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_parts
    vals = tl.load(mid + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    if DIV:
        # acc = sum(|d| - 1)  ->  mean = acc / n + 1
        acc = acc / denom + 1.0
    else:
        # acc = sum(|d| - 1)  ->  sum = acc + n
        acc = acc + denom
    tl.store(out, acc)


@triton.jit
def _l1_loss_single_kernel(
    inp,
    target,
    out,
    n_elements,
    denom,
    DIV: tl.constexpr,
    BLOCK: tl.constexpr,
    ITER: tl.constexpr,
):
    """Whole reduction in one program; masked because ``n`` is not a multiple."""
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(ITER):
        offsets = i * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        inp_val = tl.load(inp + offsets, mask=mask, other=0.0)
        target_val = tl.load(target + offsets, mask=mask, other=0.0)
        diff = tl.abs(inp_val - target_val).to(tl.float32) - 1.0
        acc += tl.where(mask, diff, 0.0)
    total = tl.sum(acc, axis=0)
    if DIV:
        total = total / denom + 1.0
    else:
        total = total + denom
    tl.store(out, total)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_input(input, target):
    if input.device != target.device:
        raise AssertionError("l1_loss: input and target must be on the same device.")
    input, target = torch.broadcast_tensors(input, target)
    return input.contiguous(), target.contiguous()


class Reduction:
    NONE = 0
    MEAN = 1
    SUM = 2


def _pad_up(p, mult):
    return -(-p // mult) * mult


def _fast_cfg(n):
    """Pick (BLOCK, ITER) for the unmasked wide-tile kernel, or None.

    ``BLOCK`` is the largest safe reduce width dividing ``n``; ``ITER`` is the
    largest power of two up to ``_FAST_ITER_MAX`` that divides ``n // BLOCK``
    so every program owns an integral number of chunks.
    """
    for blk in _FAST_BLOCKS:
        if n % blk == 0:
            m = n // blk
            it = min(_FAST_ITER_MAX, m & -m)
            return blk, it
    return None


def _l1_loss_reduce_parts(mid, n_parts, out, n_elements, reduction, dev):
    """Fold the partial buffer down to the scalar result."""
    cur, cur_p = mid, n_parts
    # Fold in bounded, mask-free stages so the reduce width never enters the
    # miscompiled regime.
    while cur_p > _FOLD:
        q = triton.cdiv(cur_p, _FOLD)
        nxt_len = _pad_up(q, _FOLD)
        nxt = torch.zeros((nxt_len,), device=dev, dtype=torch.float32)
        _l1_loss_fold_kernel[(q,)](cur, nxt, BLOCK=_FOLD)
        cur, cur_p = nxt, q
    _l1_loss_final_kernel[(1,)](
        cur,
        out,
        float(n_elements),
        cur_p,
        DIV=(reduction == Reduction.MEAN),
        BLOCK=_FOLD,
    )


def l1_loss(input: torch.Tensor, target: torch.Tensor, reduction=Reduction.MEAN):
    logger.debug("GEMS L1_LOSS (kunlunxin)")
    reduction = _normalize_reduction(reduction)
    input, target = _check_input(input, target)
    n_elements = input.numel()

    if reduction == Reduction.NONE:
        return _l1_loss_elementwise_kernel(input, target)

    if n_elements == 0:
        if reduction == Reduction.SUM:
            return torch.zeros((), device=input.device, dtype=input.dtype)
        return torch.full((), float("nan"), device=input.device, dtype=input.dtype)

    dev = input.device
    div = reduction == Reduction.MEAN
    out = torch.empty((), device=dev, dtype=input.dtype)

    with torch_device_fn.device(dev):
        cfg = _fast_cfg(n_elements) if n_elements >= _FOLD else None
        if cfg is not None:
            blk, it = cfg
            n_parts = n_elements // (blk * it)
            if n_parts == 1:
                _l1_loss_part_kernel[(1,)](
                    input,
                    target,
                    out,
                    float(n_elements),
                    DIV=div,
                    FINAL=True,
                    BLOCK=blk,
                    ITER=it,
                    num_warps=_FAST_WARPS,
                )
            else:
                mid = torch.zeros(
                    (_pad_up(n_parts, _FOLD),), device=dev, dtype=torch.float32
                )
                _l1_loss_part_kernel[(n_parts,)](
                    input,
                    target,
                    mid,
                    float(n_elements),
                    DIV=div,
                    FINAL=False,
                    BLOCK=blk,
                    ITER=it,
                    num_warps=_FAST_WARPS,
                )
                _l1_loss_reduce_parts(mid, n_parts, out, n_elements, reduction, dev)
        elif n_elements <= _SMALL_MAX:
            blk = min(_BLOCK * 8, triton.next_power_of_2(max(n_elements, 1)))
            _l1_loss_single_kernel[(1,)](
                input,
                target,
                out,
                n_elements,
                float(n_elements),
                DIV=div,
                BLOCK=blk,
                ITER=triton.cdiv(n_elements, blk),
                num_warps=_FAST_WARPS,
            )
        else:
            n_parts = triton.cdiv(n_elements, _BLOCK)
            mid = torch.zeros(
                (_pad_up(n_parts, _FOLD),), device=dev, dtype=torch.float32
            )
            _l1_loss_partial_sum_kernel[(n_parts,)](
                input, target, mid, n_elements, BLOCK=_BLOCK
            )
            _l1_loss_reduce_parts(mid, n_parts, out, n_elements, reduction, dev)
    return out
