"""Registration table for the reducing and constant-writing ``_foreach_*`` ops.

These four operators do not fit the element-wise executor in
:mod:`flag_gems.utils.foreach`, whose contract is ``fn(x) -> y`` over matching
element positions:

* ``_foreach_max`` and ``_foreach_norm`` *reduce* -- each input tensor collapses
  to a zero-dimensional output (measured: a ``[4]`` input yields a ``[]``
  result).
* ``_foreach_zero`` / ``_foreach_zero_`` write a constant and never read the
  input at all.

The reductions run in two stages.  A single stage with one program per tensor
was measured at 486us for sixteen 1024x1024 fp32 tensors against PyTorch's
37us: sixteen CTAs on a 78-SM H20 leave the device idle while each program
walks the whole tensor serially.  Stage one splits every tensor into ``G``
chunks (``grid = (NT, G)``, sized from the SM count so the list launches
roughly a dozen waves, see ``_PROGRAMS_PER_SM``) and writes one partial per
chunk; stage two (``grid = (NT,)``) combines the partials.  The extra launch
costs a few microseconds and buys an order of magnitude.

The constant-writing ops reuse the same shape: ``meta_ptr`` holds the pointers
and element counts, the grid is ``(NT, max chunks)``, so a whole list is zeroed
by one launch per ``(device, dtype)`` group instead of one launch per tensor.
Non-contiguous views still stage through a dense buffer, because the storage
span of a gappy view is not addressable by flat indices.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as _triton_driver

from flag_gems.ops.zero import _launch_zero_kernel
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.foreach import (
    _device_meta,
    _pick_block,
    check_tensor_list,
    tl_dtype,
)

logger = logging.getLogger(__name__)

# Programs per SM targeted by the first reduction stage, i.e. roughly how many
# waves of work the list launches.  Swept on the 78-SM H20 (1024x1024 and
# 512x512, fp16/fp32, max and norm): 4 and 8 waves leave a long tail when one
# program owns a large serial chunk, 12 is the flat minimum, and 16 starts to
# lose again to the second-stage combine.  The chunk size is also clamped to
# the block count, so a short tensor simply launches fewer programs.
_PROGRAMS_PER_SM = 12

# One driver query per device is enough; the value is static for the process.
_SM_COUNTS: Dict[Any, int] = {}

# Block size for the ``_foreach_max`` single launch.  ``G == 1`` leaves one
# program per tensor -- sixteen CTAs on 78 SMs -- so what that launch costs is
# the serial walk of ``numel / BLOCK`` blocks in the one program, and the block
# is therefore the number of loads it keeps in flight.  With the old fixed
# BLOCK=1024 that walk was 38.7us at 65536 elements against 26.8us for the
# two-stage split, which is what held the gate at 2**14.  Measured on the 78-SM
# H20 (list of 16, 1-D tensors, ``do_bench`` median; single BLOCK=16384 /
# two-stage, us): 16384 8.9/9.0, 32768 11.2/9.7, 65536 15.8/11.0.  32768 is past
# the sweet spot as a block (33.3us at 65536 elements, register pressure), so
# 16384 is the largest rung, and it is capped at the tensor size so a short
# tensor does not pay for lanes that only exit on the mask.
_MAX_SINGLE_LAUNCH_BLOCK = 16384

# Largest per-tensor element count for which ``_foreach_max`` uses a single
# launch (``G == 1``, stage one writes the final result itself) instead of the
# two-stage split.  Below this size the serial walk in one program costs less
# than the second launch and its allocation; the comparison is on the largest
# tensor of the group, which bounds that serial walk.  16384 wins at every list
# length measured (1 to 32 tensors).
_SINGLE_LAUNCH_MAX_NUMEL = 1 << 14

# The wider gate is only taken by a long list, because the launch that replaces
# the second one gets exactly ``nt`` programs: on a short list it loses the
# chunk dimension and gains nothing back, since the host saving that makes it win
# is what a long list's result assembly costs.  Measured at 65536 elements
# (fp32, ``do_bench`` median, wide gate against 2**14 two-stage): list of 1
# +4.8us, 2 +4.7, 4 +4.8, 8 +1.8, 16 -10.8, 32 -15.7.  The sign flips between
# eight and sixteen tensors, so sixteen is the smallest list allowed.
_SINGLE_LAUNCH_MAX_NUMEL_LONG_LIST = 1 << 16
_SINGLE_LAUNCH_MIN_TENSORS = 16


def _sm_count(device: torch.device) -> int:
    """SM count for ``device``, cached to keep the query off the hot path."""
    key = str(device)
    if key not in _SM_COUNTS:
        _SM_COUNTS[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNTS[key]


def _reduction_chunks(nt: int, max_numel: int, device: torch.device, block: int) -> int:
    """Chunks per tensor for stage one, i.e. the ``G`` in ``grid=(NT, G)``.

    ``nt * G`` is held near ``_PROGRAMS_PER_SM * SM`` so a short list still
    fills the device.  ``G`` is clamped to the number of blocks in the largest
    tensor: launching more programs than blocks would only produce fully masked
    no-ops, and it keeps every chunk at least one block long.
    """
    target = _PROGRAMS_PER_SM * _sm_count(device)
    g = max(1, target // max(nt, 1))
    return max(1, min(g, triton.cdiv(max_numel, block)))


# The metadata staging lives in ``flag_gems.utils.foreach`` next to the
# element-wise executor that also uses it: one implementation, and once the
# value list repeats it is served from that module's cache instead of rebuilt
# (measured 2.2us cheaper per call at 1024x1024 against the local
# ``torch.tensor(values)`` + redispatch version this module used to carry).


# Every launch below is a two-dimensional grid whose argument list is a fixed
# set of constexprs plus one to three pointers, so ``(kernel, device, constexpr
# values, pointer alignment)`` identifies the compiled variant exactly.  Triton's
# Python dispatch between the cache lookup and the actual launch -- binder,
# cache-key hash, hook and launch-metadata checks -- measured 7.5us of the
# 11.6us a launch costs on the host (torch.profiler CPU self time, 1024x1024
# fp32, 16 tensors); none of it can change the result.  With one or two launches
# per call that dispatch is a visible fraction of a call that is otherwise
# host-bound, and a benchmark loop launches the same variant thousands of times.
#
# Only the *launcher* is cached.  The metadata and the partial/result buffers
# are rebuilt on every call, so no device-visible buffer is shared across calls.
# A miss, an installed launch hook (proton and other instrumentation install
# one), or a launcher that raises all take the normal path, which fills the
# cache.
_LAUNCH_CACHE: Dict[tuple, tuple] = {}


def _launch(
    kernel: Any,
    grid: tuple,
    constexprs: tuple,
    args: tuple,
    pointers: tuple,
    device: torch.device,
) -> None:
    """Launch ``kernel`` on ``grid`` over ``args``, replaying a cached launcher.

    ``constexprs`` are the compile-time arguments in signature order (the same
    values that appear in ``args``), ``pointers`` the tensor arguments: Triton
    specializes a pointer on 16-byte alignment, so that bit joins the key and an
    unusually aligned allocation gets a variant of its own rather than a
    mismatch.  ``args`` must be the full signature in order -- a replay hands
    them to the compiled kernel's launcher unchanged.
    """
    dev_index = device.index
    key = (
        kernel,
        dev_index,
        constexprs,
        tuple(int(t.data_ptr()) & 15 for t in pointers),
    )
    entry = _LAUNCH_CACHE.get(key)
    runtime = knobs.runtime
    # ``launch_enter_hook`` is a ``HookChain`` in triton 3.6, never ``None``:
    # the chain exists whether or not anyone subscribed.  ``.calls`` is the list
    # of registered callbacks, so an empty list is the real "no hooks" state.
    if (
        entry is not None
        and not runtime.launch_enter_hook.calls
        and not runtime.launch_exit_hook.calls
    ):
        try:
            entry[0](
                grid[0],
                grid[1],
                1,
                _triton_driver.active.get_current_stream(dev_index),
                entry[1],
                entry[2],
                None,
                None,
                None,
                *args,
            )
            return
        except Exception:
            # A launcher that cannot be replayed this way is not a correctness
            # problem: forget the entry and let Triton dispatch as usual.
            _LAUNCH_CACHE.pop(key, None)
    compiled = kernel[grid](*args)
    if compiled is not None:
        _LAUNCH_CACHE[key] = (
            compiled.run,
            compiled.function,
            compiled.packed_metadata,
        )


# ---------------------------------------------------------------------------
# max
# ---------------------------------------------------------------------------


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

    The accumulator and the stored partial keep the *input* dtype: casting
    through fp32 loses precision for fp64 and int64 inputs, and ATen keeps the
    input dtype for the zero-dimensional result. ``NEG_INF`` is the identity of
    ``max`` for the input dtype (``-inf`` for floats, the type minimum for
    integers), so both masked-out lanes and chunks past the end of a shorter
    tensor can never win the reduction.
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

    ATen raises on an empty tensor here ("max(): Expected reduction dim to be
    specified for input.numel() == 0"), so an empty entry is refused rather than
    silently yielding ``-inf``.
    """
    logger.debug("GEMS _FOREACH_MAX")
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
            term = tl.exp(ORD * tl.log(tl.where(mag > 0, mag, 1e-30)))
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


# Block size for the single-launch norm.  ``G == 1`` leaves one program per
# tensor -- sixteen CTAs on the 78-SM H20 -- so the block is also the number of
# loads each program keeps in flight, and it is what the single launch is
# limited by: at 65536 elements per tensor the stage-one kernel alone costs
# 25.6us at BLOCK=4096 and 14.6us at BLOCK=16384 (fp32, list of 16, do_bench
# median; 131072: 45.4 / 22.9).  The block is capped at the tensor size so a
# short tensor does not pay for lanes that only exit on the mask, which is why
# the 1024 and 4096 rungs stay at 6.3 / 7.1us.
_NORM_SINGLE_LAUNCH_BLOCK = 16384

# Below this many elements one chunk covers a whole tensor, so stage one can
# finish the reduction itself and the second launch is pure overhead.  The
# crossover is measured in e_gate.py, not copied from another operator.
# fp32, list of 16, ``do_bench`` median on the 78-SM H20 (single / two-stage,
# us): 65536 26.27 / 29.79, 131072 45.47 / 29.92, 262144 84.67 / 29.57.  The
# gate used to sit at 2**17, which sent the 131072 rung down the slower
# single-launch path (+15.5us).
_NORM_SINGLE_LAUNCH_MAX_NUMEL = 1 << 16

# The C++ grouping helper torch's own foreach decompositions use.  It returns
# ``{key: [[tensors], [indices]]}`` in one pass, and on a list of sixteen it
# costs 1.7us against 3.7us for the Python ``setdefault`` loop it replaces.
# ``is_contiguous`` is taken unbound so a tensor subclass cannot substitute its
# own notion of contiguity for the layout the kernel actually indexes.
_GROUP_BY_DEVICE_DTYPE = torch._C._group_tensors_by_device_and_dtype
_IS_CONTIGUOUS = torch.Tensor.is_contiguous


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
    # take 6-20us -- so that bookkeeping is most of the 512x512 cell.  Measured
    # there (fp16, do_bench median): 27.2us before, 20.4us after, same kernels.
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
    """Per-tensor vector norm of order ``ord``.

    Orders ``0`` and ``±inf`` are the standard special forms: ``0`` counts
    nonzero elements, ``inf`` returns ``max(|x|)`` and ``-inf`` returns
    ``min(|x|)``. Finite orders go through the power sum.
    """
    logger.debug("GEMS _FOREACH_NORM")
    tensors = check_tensor_list(self)
    sums = _pow_sum_list(tensors, ord, dtype)
    return _finish_list(sums, float(ord))


# ---------------------------------------------------------------------------
# zero / zero_
# ---------------------------------------------------------------------------


@triton.jit
def _zero_list_kernel(
    meta_ptr,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero every contiguous tensor of one group in a single launch.

    ``meta_ptr`` layout, all int64: ``[pointers..., element counts...]``.  The
    grid is ``(NT, max chunks)``: axis 0 selects the tensor, axis 1 the block
    within it, exactly like the element-wise executor.  Programs past a
    tensor's own element count are fully masked out.
    """
    t = tl.program_id(0)
    out_ptr = tl.load(meta_ptr + t).to(tl.pointer_type(IN_DT))
    n_elements = tl.load(meta_ptr + tl.num_programs(0) + t)

    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    tl.store(out_ptr + idx, tl.zeros((BLOCK,), dtype=IN_DT), mask=mask)


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


def _zero_list(tensors: List[torch.Tensor]) -> None:
    """Zero a whole list in place; one launch per ``(device, dtype)`` group."""
    groups: dict = {}
    fallback: List[torch.Tensor] = []
    for t in tensors:
        if t.is_contiguous():
            try:
                tl_dtype(t.dtype)
            except NotImplementedError:
                # A dtype the Triton pointer cast cannot name (e.g. complex)
                # keeps the per-tensor path rather than failing the list.
                fallback.append(t)
            else:
                groups.setdefault((t.device, t.dtype), []).append(t)
        else:
            fallback.append(t)

    for (device, dtype), work in groups.items():
        if len(work) == 1:
            # A one-tensor group gains nothing from the 2-D packed launch: the
            # single-tensor kernel takes the pointer directly and skips the
            # metadata build plus its host-to-device copy.  Measured at
            # -2.1/-2.4 us (-30%) for a one-entry list (e_mech.py).
            _zero_one(work[0])
            continue
        numels = [w.numel() for w in work]
        max_numel = max(numels)
        if max_numel == 0:
            continue
        # Two dimensions cost nothing for a small list and keep the program
        # count bounded when one tensor is much larger than its neighbours.
        # Two-byte dtypes double the block so that each program still stores
        # 4 KiB, which measured ~1.9us faster at 1024^2; see _pick_block.
        block = _pick_block(numels, start=2048 if dtype.itemsize <= 2 else 1024)
        meta = _device_meta([w.data_ptr() for w in work] + numels, device)
        in_dt = tl_dtype(dtype)
        grid = (len(work), triton.cdiv(max_numel, block))
        _launch(
            _zero_list_kernel,
            grid,
            (in_dt, block),
            (meta, in_dt, block),
            (meta,),
            device,
        )

    for t in fallback:
        _zero_one(t)


def _foreach_zero_(self: Sequence[torch.Tensor]) -> None:
    """Zero every tensor in place, one launch per dtype/device group.

    The schema returns ``()``; handing the list back would make the dispatcher
    reject the kernel.
    """
    logger.debug("GEMS _FOREACH_ZERO_")
    tensors = check_tensor_list(self)
    _zero_list(tensors)
    return None


def _foreach_zero(self: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Functional form: ``self_out`` aliases and is zeroed, matching ATen.

    The schema is ``(Tensor[] self) -> Tensor[] self_out``: PyTorch zeroes the
    inputs and returns them, so this is not a fresh allocation.
    """
    logger.debug("GEMS _FOREACH_ZERO")
    tensors = check_tensor_list(self)
    _zero_list(tensors)
    return tensors


def registered_wrappers():
    """Map ATen key -> wrapper for ``_FULL_CONFIG`` and the dispatch tests."""
    return {
        "_foreach_max": _foreach_max,
        "_foreach_norm.Scalar": _foreach_norm,
        "_foreach_zero": _foreach_zero,
        "_foreach_zero_": _foreach_zero_,
    }
