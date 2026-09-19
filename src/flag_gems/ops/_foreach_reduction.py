"""Registration table for the reducing and constant-writing ``_foreach_*`` ops.

These four operators do not fit the element-wise executor in
:mod:`flag_gems.utils.foreach`, whose contract is ``fn(x) -> y`` over matching
element positions:

* ``_foreach_max`` and ``_foreach_norm`` *reduce* -- each input tensor collapses
  to a zero-dimensional output (measured: a ``[4]`` input yields a ``[]``
  result).
* ``_foreach_zero`` / ``_foreach_zero_`` write a constant and never read the
  input at all.

Rather than reshape the shared executor around them, each one delegates to the
Triton implementation the repository already has for the single-tensor operator
-- ``ops/max.py``, ``ops/vector_norm.py`` and ``ops/zero.py``.  That keeps the
core computation in Triton (no ATen fallback) while writing no new reduction
kernel, and it is the same "call the existing implementation" shape the
element-wise rows use when they borrow a scalar function.

The per-tensor loop here is a real cost: unlike the element-wise path, launch
count grows with list length.  It is accepted deliberately, because a fused
multi-tensor reduction would be a new kernel rather than a reuse of an existing
one, and correctness parity with ATen comes first.
"""

import logging
from typing import Any, List, Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.ops.zero import _launch_zero_kernel
from flag_gems.utils.foreach import check_tensor_list, tl_dtype

logger = logging.getLogger(__name__)


@triton.jit
def _max_kernel(
    meta_ptr,
    out_ptr,
    IN_DT: tl.constexpr,
    NEG_INF,
    BLOCK: tl.constexpr,
):
    """Per-tensor maximum over a whole TensorList, one launch for the list.

    Same metadata layout as :func:`_norm_kernel`. Looping over the list on the
    host and calling ``ops/max.py`` per tensor works and is correct, but costs a
    launch each: that version measured 2.18ms for sixty-four 512x512 tensors
    against PyTorch's 0.11ms.
    """
    t = tl.program_id(0)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)

    acc = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)
    for offset in tl.range(0, n_elements, BLOCK):
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=NEG_INF).to(tl.float32)
        acc = tl.maximum(acc, x)
    tl.store(out_ptr + t, tl.max(acc))


def _foreach_max(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Per-tensor maximum; each result is zero-dimensional.

    ATen raises on an empty tensor here ("max(): Expected reduction dim to be
    specified for input.numel() == 0"), so an empty entry is refused rather than
    silently yielding ``-inf``.
    """
    logger.debug("GEMS _FOREACH_MAX")
    tensors = check_tensor_list(self)
    for t in tensors:
        if t.numel() == 0:
            raise RuntimeError(
                "max(): Expected reduction dim to be specified for "
                "input.numel() == 0."
            )

    prepared = [t if t.is_contiguous() else t.contiguous() for t in tensors]
    results: List[Optional[torch.Tensor]] = [None] * len(prepared)
    groups: dict = {}
    for i, w in enumerate(prepared):
        groups.setdefault((w.device, w.dtype), []).append(i)

    for (device, dtype), idxs in groups.items():
        work = [prepared[i] for i in idxs]
        acc = torch.empty(len(work), dtype=torch.float32, device=device)
        meta = torch.tensor(
            [w.data_ptr() for w in work] + [w.numel() for w in work],
            dtype=torch.int64,
        ).to(device, non_blocking=True)
        neg_inf = (
            float("-inf") if dtype.is_floating_point else float(torch.iinfo(dtype).min)
        )
        _max_kernel[(len(work),)](
            meta, acc, IN_DT=tl_dtype(dtype), NEG_INF=neg_inf, BLOCK=1024
        )
        for slot, i in enumerate(idxs):
            results[i] = acc[slot].to(dtype)
    return results  # type: ignore[return-value]


@triton.jit
def _norm_kernel(
    meta_ptr,
    out_ptr,
    ORD: tl.constexpr,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``sum(|x| ** ord)`` for a whole TensorList, one launch for the list.

    ``meta_ptr`` holds ``NT`` input pointers followed by ``NT`` element counts,
    all int64; program ``t`` reduces tensor ``t`` and writes ``out_ptr[t]``.
    The final root that distinguishes a norm from a bare power sum is applied
    afterwards on the host by :func:`_finish`.

    Composing this from ``abs`` / ``pow`` / ``sum`` instead would cost four
    launches *per tensor*: that version measured 3.8ms for sixteen 512x512
    tensors against PyTorch's 0.09ms, because PyTorch reduces the entire list in
    one fused pass. Reducing per program keeps the launch count at one.

    ``ORD`` is a ``constexpr`` so orders 1 and 2 compile to a plain sum or a
    multiply rather than a call to ``pow``.
    """
    t = tl.program_id(0)
    in_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for offset in tl.range(0, n_elements, BLOCK):
        idx = offset + tl.arange(0, BLOCK)
        mask = idx < n_elements
        x = tl.load(in_ptr + idx, mask=mask, other=0).to(tl.float32)
        mag = tl.abs(x)
        if ORD == 1:
            acc += mag
        elif ORD == 2:
            acc += mag * mag
        else:
            acc += tl.exp(ORD * tl.log(tl.where(mag > 0, mag, 1e-30)))
    tl.store(out_ptr + t, tl.sum(acc))


def _pow_sum_list(
    tensors: List[torch.Tensor], ord_: Any, dtype: Optional[torch.dtype]
) -> List[torch.Tensor]:
    """One launch per ``(device, dtype)`` group, mirroring the element-wise path."""
    ord_f = float(ord_)
    ord_key = 1 if ord_f == 1.0 else (2 if ord_f == 2.0 else ord_f)

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
        acc = torch.zeros(len(idxs), dtype=torch.float32, device=device)
        work = [prepared[i] for i in idxs]
        numels = [w.numel() for w in work]
        if max(numels) > 0:
            meta = torch.tensor(
                [w.data_ptr() for w in work] + numels, dtype=torch.int64
            ).to(device, non_blocking=True)
            _norm_kernel[(len(work),)](
                meta, acc, ORD=ord_key, IN_DT=tl_dtype(in_dtype), BLOCK=1024
            )
        for slot, i in enumerate(idxs):
            results[i] = acc[slot].to(in_dtype)
    return results  # type: ignore[return-value]


def _finish(total: torch.Tensor, ord_f: float) -> torch.Tensor:
    """Apply the final root that turns a power sum into a norm."""
    if ord_f == 1.0:
        return total
    return total.sqrt() if ord_f == 2.0 else total.pow(1.0 / ord_f)


def _foreach_norm(
    self: Sequence[torch.Tensor], ord=2, dtype=None
) -> List[torch.Tensor]:
    """Per-tensor vector norm of order ``ord``."""
    logger.debug("GEMS _FOREACH_NORM")
    tensors = check_tensor_list(self)
    sums = _pow_sum_list(tensors, ord, dtype)
    return [_finish(s, float(ord)) for s in sums]


def _zero_one(t: torch.Tensor) -> None:
    """Zero one tensor in place, including non-contiguous views.

    ``ops/zero.py``'s kernel asserts contiguity, but ATen's ``_foreach_zero_``
    accepts any view (a strided slice must be zeroed in the original storage).
    A non-contiguous target is therefore zeroed through a dense staging buffer
    and copied back, mirroring what the element-wise executor does for gappy
    views.
    """
    if t.is_contiguous():
        _launch_zero_kernel(t)
        return
    staged = torch.empty_like(t, memory_format=torch.contiguous_format)
    _launch_zero_kernel(staged)
    t.copy_(staged)


def _foreach_zero_(self: Sequence[torch.Tensor]) -> None:
    """Zero every tensor in place, reusing ``ops/zero.py``'s Triton kernel.

    The schema returns ``()``; handing the list back would make the dispatcher
    reject the kernel.
    """
    logger.debug("GEMS _FOREACH_ZERO_")
    tensors = check_tensor_list(self)
    for t in tensors:
        _zero_one(t)
    return None


def _foreach_zero(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Functional form: ``self_out`` aliases and is zeroed, matching ATen.

    The schema is ``(Tensor[] self) -> Tensor[] self_out``: PyTorch zeroes the
    inputs and returns them, so this is not a fresh allocation.
    """
    logger.debug("GEMS _FOREACH_ZERO")
    tensors = check_tensor_list(self)
    for t in tensors:
        _zero_one(t)
    return tensors


def registered_wrappers():
    """Map ATen key -> wrapper for ``_FULL_CONFIG`` and the dispatch tests."""
    return {
        "_foreach_max": _foreach_max,
        "_foreach_norm.Scalar": _foreach_norm,
        "_foreach_zero": _foreach_zero,
        "_foreach_zero_": _foreach_zero_,
    }
