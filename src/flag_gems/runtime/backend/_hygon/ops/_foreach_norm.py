# Copyright (c) 2024 FlagGems. All rights reserved.
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

"""``_foreach_norm.Scalar``: grouped multi-tensor vector norm.

Each input tensor collapses to a zero-dimensional scalar in the *input* dtype
(what ATen returns; an explicit ``dtype`` argument converts first).  The list is
grouped by ``(device, dtype)``. Uniform L2 uses one last-CTA reduction launch
with stream-owned scratch. Other orders and layouts retain the two-stage
grouped reduction.

Orders ``0`` and ``±inf`` are the standard special forms: ``0`` counts nonzero
elements, ``inf`` returns ``max(|x|)`` and ``-inf`` returns ``min(|x|)``.  Finite
orders go through the power sum, accumulated in fp32 (fp64 for double inputs).
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _triton_driver

from flag_gems.ops.copy import copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils.foreach import _device_meta, check_tensor_list, tl_dtype

"Native zero-copy result views; numerical work remains in Triton."
_hygon_foreach_views__NATIVE_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_hygon_foreach_views__UNBIND = torch.ops.aten.unbind.int


def _hygon_foreach_views__unbind_views(tensor):
    return _hygon_foreach_views__UNBIND.redispatch(
        _hygon_foreach_views__NATIVE_KEYSET, tensor, 0
    )


unbind_views = _hygon_foreach_views__unbind_views
logger = logging.getLogger(__name__)
_PROGRAMS_PER_SM = 12
_SM_COUNTS: Dict[Any, int] = {}
_NORM_SINGLE_LAUNCH_BLOCK = 16384
_NORM_SINGLE_LAUNCH_MAX_NUMEL = 1 << 18
_GROUP_BY_DEVICE_DTYPE = torch._C._group_tensors_by_device_and_dtype
_IS_CONTIGUOUS = torch.Tensor.is_contiguous


def _sm_count(device: torch.device) -> int:
    """SM count for ``device``, cached to keep the query off the hot path."""
    key = str(device)
    if key not in _SM_COUNTS:
        _SM_COUNTS[key] = torch_device_fn.get_device_properties(
            device
        ).multi_processor_count
    return _SM_COUNTS[key]


def _reduction_chunks(nt: int, max_numel: int, device: torch.device, block: int) -> int:
    """Chunks per tensor for stage one, i.e. the ``G`` in ``grid=(NT, G)``."""
    target = _PROGRAMS_PER_SM * _sm_count(device)
    g = max(1, target // max(nt, 1))
    return max(1, min(g, triton.cdiv(max_numel, block)))


def _launch(kernel, grid, constexprs, args, pointers, device):
    kernel[grid](*args)


@libentry()
@triton.jit
def _norm_stage1_kernel(
    meta_ptr,
    part_ptr,
    out_ptr,
    ORD: tl.constexpr,
    ROOT: tl.constexpr,
    INF_MODE: tl.constexpr,
    IN_DT: tl.constexpr,
    ACC_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Partial power sum / extremum per ``(tensor, chunk)``.

    ``ORD`` is a ``constexpr`` so orders 1 and 2 compile to a plain sum or a
    multiply.  ``INF_MODE`` selects the extremum form: ``1`` accumulates
    ``max(|x|)`` (order ``+inf``), ``-1`` accumulates ``min(|x|)``.  Order ``0``
    counts the nonzero elements.  The accumulation is fp32 (fp64 for double
    inputs); the root is applied once in stage two.
    """
    t = tl.program_id(0)
    g = tl.program_id(1)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)
    n_blocks = tl.cdiv(n_elements, BLOCK)
    per_chunk = tl.cdiv(n_blocks, G)
    first = g * per_chunk
    last = tl.minimum(first + per_chunk, n_blocks)
    if INF_MODE == -1:
        acc = tl.full((BLOCK,), float("inf"), dtype=ACC_DT)
    else:
        acc = tl.zeros((BLOCK,), dtype=ACC_DT)
    for b in tl.range(first, last):
        idx = b * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0).to(ACC_DT)
        mag = tl.abs(x)
        if INF_MODE == 1:
            acc = tl.maximum(acc, mag)
        elif INF_MODE == -1:
            acc = tl.minimum(acc, tl.where(mask, mag, float("inf")))
        elif ORD == 0:
            acc += (mag > 0).to(ACC_DT)
        elif ORD == 1:
            acc += mag
        elif ORD == 2:
            acc += mag * mag
        else:
            term = tl_extra_shim.pow(mag, tl.full((), ORD, dtype=ACC_DT))
            acc += tl.where(mask, term, 0.0)
    if INF_MODE == 1:
        part = tl.max(acc)
    elif INF_MODE == -1:
        part = tl.min(acc)
    else:
        part = tl.sum(acc)
    if G == 1:
        if ROOT:
            part = (
                tl.sqrt(part)
                if ORD == 2
                else tl_extra_shim.pow(part, tl.full((), 1.0 / ORD, dtype=ACC_DT))
            )
        part = tl.where(n_elements == 0, 0.0, part)
        tl.store(out_ptr + t, part.to(OUT_DT))
    else:
        tl.store(part_ptr + t * G + g, part)


@libentry()
@triton.jit
def _norm_stage2_kernel(
    meta_ptr,
    part_ptr,
    out_ptr,
    ORD: tl.constexpr,
    ROOT: tl.constexpr,
    INF_MODE: tl.constexpr,
    ACC_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Combine partials and apply the root once, mirroring ATen's result."""
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
        total = (
            tl.sqrt(total)
            if ORD == 2
            else tl_extra_shim.pow(total, tl.full((), 1.0 / ORD, dtype=ACC_DT))
        )
    total = tl.where(n_elements == 0, 0.0, total)
    tl.store(out_ptr + t, total.to(OUT_DT))


_L2_SCRATCH = {}
_L2_META = {}
_L2_KERNELS = {}
_L2_CHUNKS = 64
_L2_BLOCK = 2048


@libentry()
@triton.jit
def _l2_last_cta(
    Meta,
    Partial,
    Counters,
    Out,
    N: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
    DOUBLE: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    tensor = tl.program_id(0)
    chunk = tl.program_id(1)
    pointer = tl.load(Meta + tensor).to(
        tl.pointer_type(tl.float64 if DOUBLE else Out.dtype.element_ty)
    )
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    acc = tl.zeros((BLOCK,), tl.float64 if DOUBLE else tl.float32)
    for tile in range(chunk, tl.cdiv(N, BLOCK), G):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK), BLOCK)
        if N % BLOCK == 0:
            x = tl.load(pointer + offsets).to(acc.dtype)
        else:
            x = tl.load(pointer + offsets, offsets < N, 0).to(acc.dtype)
        acc += x * x
    total = tl.sum(acc, 0)
    if G == 1:
        tl.store(Out + tensor, tl.sqrt(total))
    else:
        tl.store(Partial + tensor * G + chunk, total)
        ticket = tl.atomic_add(Counters + tensor * 32, 1, sem="acq_rel")
        if ticket == G - 1:
            r = tl.arange(0, triton.next_power_of_2(G))
            values = tl.load(Partial + tensor * G + r, r < G, 0)
            tl.store(Out + tensor, tl.sqrt(tl.sum(values, 0)))
            tl.atomic_xchg(Counters + tensor * 32, 0, sem="release")


def _l2_workspace(device, dtype, count, groups, stream):
    key = (device, stream, dtype, count, groups)
    cached = _L2_SCRATCH.get(key)
    if cached is None:
        partial = torch.empty((count, groups), dtype=dtype, device=device)
        counters = torch.zeros(count * 32, dtype=torch.int32, device=device)
        cached = (partial, counters)
        if len(_L2_SCRATCH) >= 64:
            _L2_SCRATCH.pop(next(iter(_L2_SCRATCH)))
        _L2_SCRATCH[key] = cached
    return cached


def _uniform_l2_last(tensors, n, device, dtype):
    count = len(tensors)
    block = _L2_BLOCK
    groups = min(_L2_CHUNKS, triton.cdiv(n, block))
    acc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    out = tensors[0].new_empty((count,))
    stream = _triton_driver.active.get_current_stream(device.index)
    partial, counters = _l2_workspace(device, acc_dtype, count, groups, stream)
    pointers = tuple((t.data_ptr() for t in tensors))
    key = (device, stream, dtype, pointers)
    entry = _L2_META.get(key)
    if entry is None:
        aligned = all((p % (16 * tensors[0].element_size()) == 0 for p in pointers))
        meta = _device_meta(list(pointers), device)
        entry = (meta, aligned)
        if len(_L2_META) >= 64:
            _L2_META.pop(next(iter(_L2_META)))
        _L2_META[key] = entry
    meta, aligned = entry
    args = (
        meta,
        partial,
        counters,
        out,
        n,
        groups,
        block,
        dtype == torch.float64,
        aligned,
    )
    key = (device, dtype, n, groups, block, aligned)
    kernel = _L2_KERNELS.get(key)
    if kernel is None:
        compiled = _l2_last_cta[count, groups](*args, num_warps=4)
        if isinstance(compiled, tuple):
            kernel = compiled[0]
        else:
            kernel = compiled
        if kernel is not None:
            if len(_L2_KERNELS) >= 128:
                _L2_KERNELS.pop(next(iter(_L2_KERNELS)))
            _L2_KERNELS[key] = kernel
    else:
        kernel[count, groups, 1](*args, stream=stream)
    return out


def _norm_group(
    work: List[torch.Tensor],
    numels: List[int],
    device: torch.device,
    in_dtype: torch.dtype,
    ord_key: Any,
    root: bool,
    inf_mode: int,
) -> torch.Tensor:
    """Launch the norm for one ``(device, dtype)`` group; return the result buffer."""
    nt = len(work)
    max_numel = max(numels)
    if ord_key == 2 and root and (inf_mode == 0) and (min(numels) == max_numel):
        return _uniform_l2_last(work, max_numel, device, in_dtype)
    if max_numel <= _NORM_SINGLE_LAUNCH_MAX_NUMEL:
        g = 1
    else:
        g = _reduction_chunks(nt, max_numel, device, 1024)
    meta = _device_meta([w.data_ptr() for w in work] + numels, device)
    in_dt = tl_dtype(in_dtype)
    acc_dtype = tl.float64 if in_dtype == torch.float64 else tl.float32
    if g == 1:
        out = torch.empty(nt, dtype=in_dtype, device=device)
        partial = out
        block = min(
            _NORM_SINGLE_LAUNCH_BLOCK, max(1024, triton.next_power_of_2(max_numel))
        )
    else:
        partial = torch.empty(
            nt * g,
            dtype=torch.float64 if in_dtype == torch.float64 else torch.float32,
            device=device,
        )
        out = torch.empty(nt, dtype=in_dtype, device=device)
        block = 1024
    _launch(
        _norm_stage1_kernel,
        (nt, g),
        (ord_key, root, inf_mode, in_dt, acc_dtype, in_dt, g, block),
        (
            meta,
            partial,
            out,
            ord_key,
            root,
            inf_mode,
            in_dt,
            acc_dtype,
            in_dt,
            g,
            block,
        ),
        (meta, partial, out),
        device,
    )
    if g > 1:
        block_p = triton.next_power_of_2(g)
        _launch(
            _norm_stage2_kernel,
            (nt, 1),
            (ord_key, root, inf_mode, acc_dtype, in_dt, g, block_p),
            (meta, partial, out, ord_key, root, inf_mode, acc_dtype, in_dt, g, block_p),
            (meta, partial, out),
            device,
        )
    return out


def _pow_sum_list(
    tensors: List[torch.Tensor], ord_: Any, dtype: Optional[torch.dtype]
) -> List[torch.Tensor]:
    """Two-stage norm, one pair of launches per ``(device, dtype)`` group."""
    ord_f = float(ord_)
    inf_mode = 1 if ord_f == float("inf") else -1 if ord_f == float("-inf") else 0
    ord_key = 1 if ord_f == 1.0 else 2 if ord_f == 2.0 else ord_f
    root = inf_mode == 0 and ord_f not in (0.0, 1.0)
    nt = len(tensors)
    if dtype is None and nt:
        in_dtype = tensors[0].dtype
        if in_dtype.is_floating_point and all(map(_IS_CONTIGUOUS, tensors)):
            if len(_GROUP_BY_DEVICE_DTYPE([tensors], False)) == 1:
                device = tensors[0].device
                numels = [t.numel() for t in tensors]
                if 0 not in numels:
                    out = _norm_group(
                        tensors, numels, device, in_dtype, ord_key, root, inf_mode
                    )
                    return list(unbind_views(out))
    prepared = []
    for t in tensors:
        target_dtype = t.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise RuntimeError("foreach norm requires floating input/dtype")
        work = t
        if t.dtype != target_dtype or not t.is_contiguous():
            work = torch.empty(t.shape, dtype=target_dtype, device=t.device)
            copy_(work, t)
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
            out = torch.zeros(len(idxs), dtype=in_dtype, device=device)
        values = unbind_views(out)
        for slot, i in enumerate(idxs):
            results[i] = values[slot]
    return results


@torch.jit.script
def _uniform_numel(tensors: List[torch.Tensor]) -> int:
    first = tensors[0]
    n = first.numel()
    for tensor in tensors:
        if (
            tensor.dtype != first.dtype
            or tensor.device != first.device
            or tensor.numel() != n
            or (not tensor.is_contiguous())
        ):
            return -1
    return n


def _foreach_norm(
    self: Sequence[torch.Tensor], ord=2, dtype=None
) -> List[torch.Tensor]:
    """Per-tensor vector norm of order ``ord``.

    Orders ``0`` and ``±inf`` are the standard special forms: ``0`` counts
    nonzero elements, ``inf`` returns ``max(|x|)`` and ``-inf`` returns
    ``min(|x|)``.  Finite orders go through the power sum.  Each result is a
    zero-dimensional tensor in the input dtype unless ``dtype`` is given.
    """
    logger.debug("GEMS_HYGON _FOREACH_NORM")
    tensors = check_tensor_list(self)
    if ord == 2 and dtype is None and tensors[0].dtype.is_floating_point:
        n = _uniform_numel(tensors)
        if n > 0:
            out = _uniform_l2_last(tensors, n, tensors[0].device, tensors[0].dtype)
            return list(unbind_views(out))
    return _pow_sum_list(tensors, ord, dtype)


def registered_wrappers():
    """Map ATen key -> wrapper for the dispatch table."""
    return {"_foreach_norm.Scalar": _foreach_norm}
