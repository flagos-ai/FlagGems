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

"""XPU override for ``aten::_foreach_max`` (per-tensor maximum reduction).

The generic two-stage reduction in ``flag_gems.ops._foreach_reduction`` fails
to compile on the XPU backend for sub-fp32 inputs: the loop carries

    acc = tl.maximum(acc, x).to(IN_DT)

which the XPU ``TritonXPUDtypeConvert`` pass rejects with
``'scf.for' op 0-th init and 0-th region iter_arg have different type`` --
``tl.maximum`` widens sub-fp32 types, so the ``scf.for`` init (fp32) and the
loop-carried value (input dtype) disagree.  This override keeps the same
two-stage structure and the tuned constants of the generic implementation, but
accumulates fp16/bf16 in fp32 (type-constant loop, same pattern as the generic
``_norm_stage1_kernel``); ``max`` has no accumulation rounding, so the only
cost is the final cast back to the input dtype.  fp32/fp64/ints keep the input
dtype: an fp32 accumulator would lose precision for fp64 and int64 inputs, and
ATen keeps the input dtype for the zero-dimensional result.
"""

import logging
from typing import Any, List, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_reduction import (
    _GROUP_BY_DEVICE_DTYPE,
    _IS_CONTIGUOUS,
    _MAX_SINGLE_LAUNCH_BLOCK,
    _NORM_SINGLE_LAUNCH_BLOCK,
    _NORM_SINGLE_LAUNCH_MAX_NUMEL,
    _SINGLE_LAUNCH_MAX_NUMEL,
    _SINGLE_LAUNCH_MAX_NUMEL_LONG_LIST,
    _SINGLE_LAUNCH_MIN_TENSORS,
    _launch,
    _reduction_chunks,
)
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.foreach import _device_meta, check_tensor_list, tl_dtype

logger = logging.getLogger(__name__)


@triton.jit
def _max_stage1_kernel(
    meta_ptr,
    part_ptr,
    IN_DT: tl.constexpr,
    NEG_INF,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Partial max per ``(tensor, chunk)``, written to ``part_ptr[t * G + g]``.

    Sub-fp32 inputs (fp16/bf16) accumulate in fp32: ``tl.maximum`` widens
    sub-fp32 types, and the round-trip cast back to the input dtype at every
    iteration leaves the ``scf.for`` init as fp32 but the loop-carried value as
    the input dtype, which the XPU backend rejects during type conversion.
    Accumulating in fp32 keeps the loop type constant and, since ``max`` has no
    accumulation rounding, costs only the final cast -- the same pattern as the
    generic ``_norm_stage1_kernel``.  fp32/fp64/ints keep the *input* dtype: an
    fp32 accumulator would lose precision for fp64 and int64 inputs, and ATen
    keeps the input dtype for the zero-dimensional result. ``NEG_INF`` is the
    identity of ``max`` for the input dtype (``-inf`` for floats, the type
    minimum for integers), so both masked-out lanes and chunks past the end of
    a shorter tensor can never win the reduction.
    """
    t = tl.program_id(0)
    g = tl.program_id(1)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)

    # Chunk boundaries are derived from the block count, not from an element
    # count divided by G: every chunk then starts on a BLOCK boundary, so a
    # chunk is a whole number of full-width loads rather than a ragged slice.
    n_blocks = tl.cdiv(n_elements, BLOCK)
    per_chunk = tl.cdiv(n_blocks, G)
    first = g * per_chunk
    last = tl.minimum(first + per_chunk, n_blocks)

    if IN_DT == tl.float16 or IN_DT == tl.bfloat16:
        acc = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)
        for b in tl.range(first, last):
            idx = b * BLOCK + tl.arange(0, BLOCK)
            mask = idx < n_elements
            acc = tl.maximum(
                acc, tl.load(in_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
            )
        tl.store(part_ptr + t * G + g, tl.max(acc).to(IN_DT))
    else:
        acc = tl.full((BLOCK,), NEG_INF, dtype=IN_DT)
        for b in tl.range(first, last):
            idx = b * BLOCK + tl.arange(0, BLOCK)
            mask = idx < n_elements
            x = tl.load(in_ptr + idx, mask=mask, other=NEG_INF)
            # ``tl.maximum`` widens sub-fp32 types, so the result is cast back to
            # keep the loop-carried type stable and the stored value in input dtype.
            acc = tl.maximum(acc, x).to(IN_DT)
        tl.store(part_ptr + t * G + g, tl.max(acc))


@triton.jit
def _max_stage2_kernel(
    part_ptr,
    out_ptr,
    IN_DT: tl.constexpr,
    NEG_INF,
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Combine the ``G`` partials of tensor ``t`` into ``out_ptr[t]``."""
    t = tl.program_id(0)
    idx = tl.arange(0, BLOCK_P)
    partial = tl.load(part_ptr + t * G + idx, mask=idx < G, other=NEG_INF)
    tl.store(out_ptr + t, tl.max(partial).to(IN_DT))


def _foreach_max(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Per-tensor maximum; each result is zero-dimensional.

    Mirrors ``flag_gems.ops._foreach_reduction._foreach_max`` (same grouping,
    same single/two-stage gate) but refers to the stage-one kernel above, whose
    gp16/bf16 path compiles on the XPU backend.  ATen raises on an empty tensor
    here ("max(): Expected reduction dim to be specified for input.numel() ==
    0"), so an empty entry is refused rather than silently yielding ``-inf``.
    """
    logger.debug("GEMS_KUNLUNXIN _FOREACH_MAX")
    tensors = check_tensor_list(self)
    # One ``numel()`` pass feeds both the emptiness check and the metadata; the
    # list is walked several times below and every pass costs microseconds on a
    # call this short.
    numels = [t.numel() for t in tensors]
    if 0 in numels:
        raise RuntimeError(
            "max(): Expected reduction dim to be specified for " "input.numel() == 0."
        )

    prepared = [t if t.is_contiguous() else t.contiguous() for t in tensors]
    results: List[Optional[torch.Tensor]] = [None] * len(prepared)
    groups: dict = {}
    for i, w in enumerate(prepared):
        groups.setdefault((w.device, w.dtype), []).append(i)

    for (device, dtype), idxs in groups.items():
        work = [prepared[i] for i in idxs]
        work_numels = [numels[i] for i in idxs]
        nt = len(work)
        max_numel = max(work_numels)
        meta = _device_meta([w.data_ptr() for w in work] + work_numels, device)
        neg_inf = (
            float("-inf") if dtype.is_floating_point else float(torch.iinfo(dtype).min)
        )
        in_dt = tl_dtype(dtype)
        single = max_numel <= _SINGLE_LAUNCH_MAX_NUMEL or (
            max_numel <= _SINGLE_LAUNCH_MAX_NUMEL_LONG_LIST
            and nt >= _SINGLE_LAUNCH_MIN_TENSORS
        )
        if single:
            # ``G == 1`` makes the chunk the whole tensor, so stage one's
            # ``part_ptr[t * G + g]`` is ``out_ptr[t]``: the partial buffer *is*
            # the result buffer (same dtype), and no second launch is needed.
            g = 1
            block = min(
                _MAX_SINGLE_LAUNCH_BLOCK,
                max(1024, triton.next_power_of_2(max_numel)),
            )
            out = torch.empty(nt, dtype=dtype, device=device)
            _launch(
                _max_stage1_kernel,
                (nt, g),
                (in_dt, neg_inf, g, block),
                (meta, out, in_dt, neg_inf, g, block),
                (meta, out),
                device,
            )
        else:
            g = _reduction_chunks(nt, max_numel, device, 1024)
            # Two allocations rather than one buffer with two slices: a
            # ``buf[:n]``/``buf[n:]`` pair costs ~7.7us each inside ``use_gems``
            # (FlagGems registers ``slice`` and ``narrow`` with Python-level
            # implementations), an extra ``torch.empty`` about 2us.
            partial = torch.empty(nt * g, dtype=dtype, device=device)
            out = torch.empty(nt, dtype=dtype, device=device)
            _launch(
                _max_stage1_kernel,
                (nt, g),
                (in_dt, neg_inf, g, 1024),
                (meta, partial, in_dt, neg_inf, g, 1024),
                (meta, partial),
                device,
            )
            _launch(
                _max_stage2_kernel,
                (nt, 1),
                (in_dt, neg_inf, g, triton.next_power_of_2(g)),
                (partial, out, in_dt, neg_inf, g, triton.next_power_of_2(g)),
                (partial, out),
                device,
            )
        # ``unbind`` is registered by FlagGems itself, so inside ``use_gems`` it
        # costs a Python-level implementation (~7us more than the C++ select
        # this loop compiles to).  ``out[slot]`` stays on the stock kernel.
        for slot, i in enumerate(idxs):
            results[i] = out[slot]
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


@triton.jit
def _norm_stage1_kernel(
    meta_ptr,
    part_ptr,
    out_ptr,
    ORD: tl.constexpr,
    ROOT: tl.constexpr,
    INF_MODE: tl.constexpr,
    IN_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Partial power sum / extremum per ``(tensor, chunk)``.

    Mirrors ``flag_gems.ops._foreach_reduction._norm_stage1_kernel``; the only
    difference is the generic ``tl.where(mag > 0, mag, 1e-30)``, whose
    ``v16i1 = setcc v16f32, splat(0.0)`` the XPU ``llc`` cannot select
    (``LLVM ERROR: Cannot select``) for orders outside 0/1/2/±inf.  The
    ``tl.maximum`` form keeps the same value for every ``mag`` outside the
    ``(0, 1e-30)`` interval (``maxf`` returns the non-NaN operand) and is the
    identity-free form the ``INF_MODE == 1`` path already uses.

    ``ORD`` is a ``constexpr`` so orders 1 and 2 compile to a plain sum or a
    multiply rather than a call to ``pow``.  ``INF_MODE`` selects the extremum
    form of the norm: ``1`` accumulates ``max(|x|)`` (order ``+inf``) and ``-1``
    accumulates ``min(|x|)`` (order ``-inf``).  Order ``0`` counts the nonzero
    elements, which is the limit of ``sum |x|^p`` as ``p -> 0``.

    The accumulation is fp32 for accuracy.  The root is *not* applied here: it
    is applied once in the second stage, after the partials are combined.
    """
    t = tl.program_id(0)
    g = tl.program_id(1)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)

    n_blocks = tl.cdiv(n_elements, BLOCK)
    per_chunk = tl.cdiv(n_blocks, G)
    first = g * per_chunk
    last = tl.minimum(first + per_chunk, n_blocks)

    # ``min`` needs the true identity (+inf), not zero: zero would clamp every
    # positive chunk to 0.  ``max`` and the sums use 0, which is the identity
    # there and also lets chunks past the end of a shorter tensor drop out.
    if INF_MODE == -1:
        acc = tl.full((BLOCK,), float("inf"), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in tl.range(first, last):
        idx = b * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0).to(tl.float32)
        mag = tl.abs(x)
        if INF_MODE == 1:
            acc = tl.maximum(acc, mag)
        elif INF_MODE == -1:
            acc = tl.minimum(acc, tl.where(mask, mag, float("inf")))
        elif ORD == 0:
            acc += (mag > 0).to(tl.float32)
        elif ORD == 1:
            acc += mag
        elif ORD == 2:
            acc += mag * mag
        else:
            term = tl.exp(ORD * tl.log(tl.maximum(mag, 1e-30)))
            acc += tl.where(mask, term, 0.0)
    if INF_MODE == 1:
        part = tl.max(acc)
    elif INF_MODE == -1:
        part = tl.min(acc)
    else:
        part = tl.sum(acc)
    if G == 1:
        # This program owns every chunk of the tensor, so the second stage
        # would only re-combine a single partial: finish here instead.
        if ROOT:
            part = tl.sqrt(part) if ORD == 2 else tl_extra_shim.pow(part, 1.0 / ORD)
        part = tl.where(n_elements == 0, 0.0, part)
        tl.store(out_ptr + t, part.to(OUT_DT))
    else:
        tl.store(part_ptr + t * G + g, part)


@triton.jit
def _norm_stage2_kernel(
    meta_ptr,
    part_ptr,
    out_ptr,
    ORD: tl.constexpr,
    ROOT: tl.constexpr,
    INF_MODE: tl.constexpr,
    OUT_DT: tl.constexpr,
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Combine partials and apply the root once, mirroring ATen's result.

    ``ROOT`` compiles away the final root for the orders that have none.  The
    stored value is cast back to ``OUT_DT``, which is the input dtype -- the
    dtype ATen returns.

    A zero-element tensor has no chunks; ``ATen`` returns 0 for the orders that
    have an identity, and this operator has always returned 0 (rather than
    leaving ``torch.empty`` garbage) for an empty entry, so that is made
    explicit here.
    """
    t = tl.program_id(0)
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)

    idx = tl.arange(0, BLOCK_P)
    if INF_MODE == -1:
        other = float("inf")
    else:
        other = 0.0
    partial = tl.load(part_ptr + t * G + idx, mask=idx < G, other=other)
    if INF_MODE == 1:
        total = tl.max(partial)
    elif INF_MODE == -1:
        total = tl.min(partial)
    else:
        total = tl.sum(partial)
    if ROOT:
        total = tl.sqrt(total) if ORD == 2 else tl_extra_shim.pow(total, 1.0 / ORD)
    total = tl.where(n_elements == 0, 0.0, total)
    tl.store(out_ptr + t, total.to(OUT_DT))


def _norm_group(
    work: List[torch.Tensor],
    numels: List[int],
    device: torch.device,
    in_dtype: torch.dtype,
    ord_key: Any,
    root: bool,
    inf_mode: int,
) -> torch.Tensor:
    """Launch the norm for one ``(device, dtype)`` group; return the result buffer.

    ``work`` is non-empty and at least one entry has elements.  The result
    buffer is ``len(work)`` values in the input dtype, one per tensor.
    """
    nt = len(work)
    max_numel = max(numels)
    if max_numel <= _NORM_SINGLE_LAUNCH_MAX_NUMEL:
        g = 1
    else:
        g = _reduction_chunks(nt, max_numel, device, 1024)
    meta = _device_meta([w.data_ptr() for w in work] + numels, device)
    in_dt = tl_dtype(in_dtype)
    if g == 1:
        # ``G == 1`` makes the chunk the whole tensor, and stage one then stores
        # straight to ``out_ptr``: the partial pointer is never dereferenced, so
        # the fp32 partial buffer the two-stage path needs is dead weight here
        # (~2us of host time for the allocation, on a host-bound call).  Hand
        # ``out`` to that slot the way ``_foreach_max`` already does.
        out = torch.empty(nt, dtype=in_dtype, device=device)
        partial = out
        block = min(
            _NORM_SINGLE_LAUNCH_BLOCK, max(1024, triton.next_power_of_2(max_numel))
        )
    else:
        # fp32 partials keep the accumulation accuracy of the single-stage
        # kernel even though the output dtype may be fp16/bf16.
        partial = torch.empty(nt * g, dtype=torch.float32, device=device)
        out = torch.empty(nt, dtype=in_dtype, device=device)
        block = 1024
    _launch(
        _norm_stage1_kernel,
        (nt, g),
        (ord_key, root, inf_mode, in_dt, in_dt, g, block),
        (meta, partial, out, ord_key, root, inf_mode, in_dt, in_dt, g, block),
        (meta, partial, out),
        device,
    )
    if g > 1:
        block_p = triton.next_power_of_2(g)
        _launch(
            _norm_stage2_kernel,
            (nt, 1),
            (ord_key, root, inf_mode, in_dt, g, block_p),
            (meta, partial, out, ord_key, root, inf_mode, in_dt, g, block_p),
            (meta, partial, out),
            device,
        )
    return out


def _pow_sum_list(
    tensors: List[torch.Tensor], ord_: Any, dtype: Optional[torch.dtype]
) -> List[torch.Tensor]:
    """Two-stage norm, one pair of launches per ``(device, dtype)`` group."""
    ord_f = float(ord_)
    inf_mode = 1 if ord_f == float("inf") else (-1 if ord_f == float("-inf") else 0)
    ord_key = 1 if ord_f == 1.0 else (2 if ord_f == 2.0 else ord_f)
    # Orders 0/1/±inf accumulate the norm directly; the rest need the root.
    root = inf_mode == 0 and ord_f not in (0.0, 1.0)

    # Fast path for the shape a foreach call actually has: no dtype conversion
    # requested, every entry already contiguous and floating point, and a single
    # ``(device, dtype)`` group.  The generic path below then walks the list four
    # times (convert, group, rebuild the per-group work/numel lists, assemble the
    # views) and its Python dict grouping costs twice the C++ group.  These calls
    # are host-bound instead of kernel-bound -- the scored latency tracks host
    # time minus a constant ~30us at every cell, list of 16, while the kernels
    # take 6-20us -- so that bookkeeping is most of the 512x512 cell.
    nt = len(tensors)
    if dtype is None and nt:
        in_dtype = tensors[0].dtype
        if in_dtype.is_floating_point and all(map(_IS_CONTIGUOUS, tensors)):
            # One C++ pass answers both "same device" and "same dtype".
            if len(_GROUP_BY_DEVICE_DTYPE([tensors], False)) == 1:
                device = tensors[0].device
                numels = [t.numel() for t in tensors]
                if 0 not in numels:
                    out = _norm_group(
                        tensors, numels, device, in_dtype, ord_key, root, inf_mode
                    )
                    # Plain ``select``: ``unbind`` is registered by FlagGems
                    # itself, so inside ``use_gems`` it costs ~7us more.
                    return [out[i] for i in range(nt)]

    prepared = []
    for t in tensors:
        work = t if dtype is None else t.to(dtype)
        if not work.is_contiguous():
            work = work.contiguous()
        if not work.dtype.is_floating_point:
            work = work.to(torch.float32)
        prepared.append(work)

    results: List[Optional[torch.Tensor]] = [None] * len(prepared)
    groups: dict = {}
    for i, w in enumerate(prepared):
        groups.setdefault((w.device, w.dtype), []).append(i)

    for (device, in_dtype), idxs in groups.items():
        work = [prepared[i] for i in idxs]
        numels = [w.numel() for w in work]
        if max(numels) > 0:
            out = _norm_group(work, numels, device, in_dtype, ord_key, root, inf_mode)
        else:
            # Every entry is empty; the stage-two kernel would not be launched,
            # so initialize the results instead of returning uninitialized memory.
            out = torch.zeros(len(idxs), dtype=in_dtype, device=device)
        # Same reason as in ``_foreach_max``: plain ``select`` avoids the
        # Python-level ``unbind`` that FlagGems registers while gems are on.
        for slot, i in enumerate(idxs):
            results[i] = out[slot]
    return results  # type: ignore[return-value]


def _finish_list(sums: List[torch.Tensor], ord_f: float) -> List[torch.Tensor]:
    """The norm and its root are both applied inside the kernels."""
    return sums


def _foreach_norm(
    self: Sequence[torch.Tensor], ord=2, dtype=None
) -> List[torch.Tensor]:
    """Per-tensor vector norm of order ``ord`` (XPU override).

    Orders ``0`` and ``±inf`` are the standard special forms: ``0`` counts
    nonzero elements, ``inf`` returns ``max(|x|)`` and ``-inf`` returns
    ``min(|x|)``. Finite orders go through the power sum.
    """
    logger.debug("GEMS_KUNLUNXIN _FOREACH_NORM")
    tensors = check_tensor_list(self)
    sums = _pow_sum_list(tensors, ord, dtype)
    return _finish_list(sums, float(ord))


# ``_foreach_norm.Scalar`` resolves to ``_foreach_norm_Scalar`` in the
# SpecOpRegistrar-based overlay (tests/test_foreach_ops.py), so the same
# function is exported under both spellings.
_foreach_norm_Scalar = _foreach_norm


def registered_wrappers():
    """Map ATen key -> wrapper (mirrors the generic module's API)."""
    return {
        "_foreach_max": _foreach_max,
        "_foreach_norm.Scalar": _foreach_norm,
    }


__all__ = ["_foreach_max", "_foreach_norm"]
