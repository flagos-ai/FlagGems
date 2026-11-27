"""In-place elementwise bitwise XOR (``A ^= B``) for Ascend NPU, written in TLE.

Semantics of the reference (FlagGems ``bitwise_xor_tensor_``): every element of
``A`` is replaced by ``A[i] ^ B[i]`` and the **same tensor object** is returned,
so the caller observes ``out is A`` with ``A`` mutated in place.

Kernel shape: **vec-explicit** (from the TLE shape decision tree -- the operator
has no ``tl.dot``, needs no cross-engine handshake and is pure streaming, so the
whole story is explicit UB staging):

    GM -> (MTE2, ``tle.dsa.copy``) -> UB -> ``triton.language`` ``^`` -> GM

XOR has no member in the TLE elementwise set, so the compute itself is plain
``triton.language``, as the implementation profile requires; every eligible bulk
tile move goes through ``tle.dsa.copy``.

Two launches, and why the staged kernel has no branch
-----------------------------------------------------
``tle.dsa.copy`` takes an explicit shape but **no mask**, so a tensor whose
length is not a multiple of the tile width needs a masked tail.  Round 3 kept
that tail *inside* the staged kernel as an ``if base + BLOCK <= n: ... else:
...`` branch, and the round-3 compiler IR showed the price: the compiler
allocated UB for the buffers of *both* sides of the branch and applied its
automatic double buffering (``hivm.multi_buffer = 2``) to all six of them, for an
180224-byte claim inside the 192 KB UB -- including the three buffers that belong
only to the masked side, which is entered at most once per program and never at
all in any measured timing shape.

So the ragged remainder is handled by a **separate masked kernel** launched over
``n_full .. n``, and the staged kernel is branch-free: it is launched only over
the full-tile prefix ``n_full = (n // block) * block`` and its body carries no
conditional and no mask.  Inputs shorter than ``_SMALL_LIMIT`` skip the staged
kernel entirely and go straight to the masked kernel, which is what keeps the
64x64 cases (4096 elements) off a wide mask.  Measured effect of the split at an
unchanged 16384-byte payload: the double buffering is retained and the L2-resident
latencies reproduce the branch-carrying round-3 form to within 0.6%, so the
branch removal is free; the 2**30 cells come out 1.7% (bool) and 0.2% (int16)
faster than round 3.

Why the tile is sized in bytes and not in elements
--------------------------------------------------
Round-1 profiling (msprof, CANN 9.0.0) showed the kernel losing on the narrow
element types with MTE2 as the busiest pipe (duty ~0.905 at 2**30) while VEC sat
at 0.16-0.19, so the GM->UB staging path -- not the vector op -- sets the cost.
Round-3 profiling confirmed that ordering on the then-current source (MTE2 duty
0.87-0.90 against VEC 0.17-0.19 and MTE3 0.21-0.42) and added the decisive fact:
a larger staged payload still pays even at constant total bytes.  Going from a
4 KB to a 16 KB payload at an unchanged byte count cut the 2**30 bool case from
4.636087 to 2.878829 ms (-38%) and the 2**30 int16 case from 6.483386 to
5.702362 ms (-12%), while int32 -- whose payload was already 16 KB and therefore
did not change -- stayed flat.  A pure bandwidth wall forbids that, so a
copy-count-dependent term of the form ``t = bytes/W + copies*F`` with
``F ~ 0.04-0.06 us`` per staged copy was still being paid at 16 KB.

The tile is therefore chosen from a *payload byte* target: each of the two staged
GM->UB copies moves ~``payload`` bytes.  Round 2 introduced the byte target but an
element clamp truncated it (uint8 4 KB, int16 8 KB); round 3 lifted that and
reached 16 KB at every width; round 4 doubled it to 32 KB.

Why the payload is dispatched on L2 residency
---------------------------------------------
The payload cannot be one constant, because the two workload regimes want
opposite things, and the round-4/round-5 pair isolates the reason.

The compiler's live UB set is three buffers: the two staging buffers plus the
value ``tl.store`` needs (the UB->GM direction of ``tle.dsa.copy`` does not lower
on this toolchain -- see below).  At a 16384-byte payload that is
``3 x 2 x 16384 = 98304 B`` of the 192 KB per-core UB with the automatic double
buffering retained -- three ``hivm.multi_buffer = 2`` marks, no UB-overflow note
(round 5, confirmed from the captured NPUIR).  At 32768 bytes the doubled
footprint would be 196608 B, so the compiler reports ``[NOTE] Ub overflow
detected; automatically set enable-auto-multi-buffer to false`` and drops the
marks entirely (round 4).  The UB budget, not the claim size, is binding: both
payloads claim 98304 B, one double buffered and one not.

Which of the two is faster depends on whether the copies wait on DRAM.

  * Footprint not in L2 (the two 2**30 shapes, and the int32 2**24 case at
    192 MiB > 168 MiB): the copies wait on DRAM, so the MTE2/VEC overlap is what
    keeps MTE2 busy -- at 2**30 it is genuinely bandwidth-ceilinged, MTE2 duty
    0.8905 at a uniform ~1.13 TB/s for every width.  The 16384-byte
    multi-buffered payload wins there (bool::0 2.831153 against round 4's
    2.821451 -- inside the per-cell noise floor -- and int32::2/::3 0.175745 /
    0.176071 against round 4's 0.1871, i.e. 6%).
  * Footprint in L2 (bool and int16 at 2**24: 48 MiB and 96 MiB of 168 MiB): no
    pipe is saturated (round-5 profile: MTE2 0.5672/0.5943, VEC only marginally
    higher at 0.6056/0.6316, MTE3 ~0.44), so these cells are limited by per-copy
    overhead and dependency serialization, and there the 32768-byte payload --
    half the iterations per program -- is 8.6-12.3% faster (bool::2/::3
    0.012716/0.012705, int16::2/::3 0.022337/0.022238, against 0.013915/0.013919
    and 0.025445/0.025440 at 16384 B).

So the payload is selected at run time from ``3 * n * itemsize <= L2``: the wide
payload for a footprint the cache can hold, the narrow multi-buffered one
otherwise.  ``int32`` at 2**24 makes the rule self-consistent -- at 192 MiB it
misses the cache and indeed prefers the narrow payload, exactly like the 2**30
cells.  The L2 capacity is read from the live device and never hard-coded: the
submission target is Ascend910B2, which need not be the SKU the development
service reports, and a failed query simply keeps the narrow payload.

Why the write-back is ``tl.store`` and not ``tle.dsa.copy``
-----------------------------------------------------------
On this toolchain (CANN 9.0.0, FlagTree ascend3.5 line) the UB->GM direction of
``tle.dsa.copy`` does not lower:

  * source is a register tensor -> the TileIR ``tile.copy`` becomes a
    ``hivm.hir`` load of the output, rejected by BiShengIR
    (``hivm.hir.load op only support src == gm and dst != gm``);
  * source is a UB buffer -> ``TritonToLinalgIncubated`` dies with
    ``failed to legalize unresolved materialization from
    ('memref<1xi16, strided<[1], offset: ?>>') to ('!tt.ptr<i16>')``.

Routing the result back into ``a_ub`` so the store reads a *buffer* does not
avoid it either: both ``tle.dsa.to_buffer`` and a tensor->buffer ``tle.dsa.copy``
fail in the same ``TritonToLinalgIncubated`` pass (probe V0-V3), and
``tle.dsa.insert_slice`` is defined over tensors, not buffers.  The GM->UB
direction does have a legal form: a ``tl.make_block_ptr`` source (the pointer is
materialised inside the block pointer rather than as a bare ``tt.addptr``), which
is what this kernel uses.  ``tl.store`` is therefore the documented escape hatch
for the one access DSA cannot express, and it is the only non-DSA movement in the
kernel -- at the cost of the third live UB value that bounds the payload above.

Why one kernel per element count and not one per shape
------------------------------------------------------
The op is bit granular: XOR is independent per bit, so the payload can be staged
under any same-width integer type without changing a single output bit.  The
kernel is written against ``element_ty`` and stages whatever type the pointers
carry, which keeps ``int16``/``int32`` on native vector lanes.  ``torch.bool``
is the single exception -- Triton models it as ``i1``, which has no UB buffer
form -- so the wrapper hands the kernel a zero-copy ``uint8`` view (one byte per
element, 0/1 valued, so byte-wise XOR is exactly the logical XOR).

This is *not* expressed with ``@triton.autotune``: this operator mutates ``A`` in
place and XOR is not idempotent, so a tuner that benchmarks a configuration by
re-launching the kernel would XOR ``A`` with ``B`` once per benchmarking
repetition and corrupt the very result the harness checks afterwards.  The tile
is therefore a deterministic host-side dispatch on ``(n, element size, L2)``.

Grid and portability
--------------------
The submission target is Ascend910B2 while the development/evaluation service
may report a different 910B SKU, so both the core count and the L2 capacity are
queried from the live device at run time and never hard-coded; the grid is capped
both by that core count and by the device's 1-D grid limit, and the tiles a
program cannot reach in one pass are picked up by an interleaved strided loop
(``pid``, ``pid + nprog``, ...).  This is what keeps the 2**30-element cases
legal: their tile count exceeds the maximum 1-D grid extent, so a
single-tile-per-program launch could not cover them.
"""

from __future__ import annotations

import logging

import torch
import torch_npu  # noqa: F401  (must precede triton on this stack)
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)

# Bytes moved by one staged GM->UB copy.  The per-copy fixed cost of
# ``tle.dsa.copy`` (~0.04-0.06 us measured at 2**30) is amortised over this many
# bytes, and it is also the UB budget knob: the compiler's live UB set is three
# buffers, so ``3 x 2 x payload`` must fit the per-core UB for the automatic
# double buffering (``hivm.multi_buffer = 2``) to survive.
#
#   * _PAYLOAD_WIDE = 32768 -> ``3 x 32768 = 98304 B`` single buffered after the
#     compiler reports "Ub overflow detected".  Measured 8.6-12.3% faster on the
#     cells whose footprint fits L2, where the copies do not wait on DRAM and the
#     halved iteration count is what counts.
#   * _PAYLOAD_NARROW = 16384 -> ``3 x 2 x 16384 = 98304 B`` with double
#     buffering retained (three ``hivm.multi_buffer = 2`` marks, no overflow
#     note).  The choice whenever the footprint misses the cache, where the
#     MTE2/VEC overlap is what keeps the bandwidth-bound stream busy.
_PAYLOAD_NARROW = 16384
_PAYLOAD_WIDE = 32768

# Smallest tile tolerated, so short tensors still get a usable vector width.
_BLOCK_MIN = 1024

# Largest tile tolerated, i.e. the wide payload divided by the narrowest element
# (uint8).  Round 2 measured the fixed per-copy staging term on top of a ~16 KB
# payload: at a 4096-element clamp the uint8 copy moved only 4 KB and the int16
# copy 8 KB, so the byte target was not reached and those two types still lost.
# At the wide payload every width stages its full payload (uint8 32768, int16
# 16384, int32 8192 elements); at the narrow payload the clamp is not reached at
# all (uint8 16384 elements exactly).
_BLOCK_MAX = 32768

# Element counts at or below this are dominated by launch cost and are too small
# to fill a wide tile, so they go straight to the masked kernel with the narrow
# proven tile rather than paying a wide mask over a fraction of the work: a wide
# tile over a tiny tensor would push the whole problem through the masked path,
# paying a full wide mask and a wide UB allocation for a fraction of the work
# (measured as 0.708x for a 4096-element int32 tensor in the sibling out-of-place
# operator, which is why that dispatch exists there too).
_SMALL_LIMIT = 8192
_BLOCK_SMALL = 2048

# Ascend allows at most 65535 programs per grid dimension; used only when the
# live vector-core count cannot be read.
_MAX_GRID_1D = 65535

# Programs wanted per vector core.  A small multiple keeps several independent
# MTE2/MTE3 streams in flight per core without letting the grid run away on the
# large shapes.
_PROGRAMS_PER_CORE = 8

_NUM_VEC_CORES = None
_L2_BYTES = {}


def _num_vector_cores():
    """Live vector-core count of the *actual* target device (0 when unknown).

    Never hard-code a core count observed on the development device: the
    submission target is Ascend910B2, which need not be the SKU the service
    reports.  This is output-independent target state, so it is resolved once and
    reused; when the target will not report it the caller falls back to the
    device's maximum 1-D grid extent.
    """
    global _NUM_VEC_CORES
    if _NUM_VEC_CORES is not None:
        return _NUM_VEC_CORES

    count = 0
    try:
        from triton.backends.ascend import driver as _ascend_driver

        props = _ascend_driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        if isinstance(props, dict):
            count = props.get("num_vectorcore") or props.get("vector_core_num") or 0
        else:
            count = getattr(props, "num_vectorcore", 0) or getattr(
                props, "vector_core_num", 0
            )
    except Exception:
        count = 0

    if not count:
        try:
            count = int(
                torch.npu.get_device_properties(
                    torch.npu.current_device()
                ).vector_core_num
            )
        except Exception:
            count = 0

    _NUM_VEC_CORES = int(count) if count else 0
    return _NUM_VEC_CORES


def _l2_bytes(device_index):
    """L2 capacity of the live device, or 0 when it cannot be read.

    Same portability rule as the core count: read it, never assume it.  On this
    stack the Triton Ascend driver exposes no ``active`` handle at all, so the
    query goes through ``torch.npu``, whose ``_NPUDeviceProperties`` carries
    ``L2_cache_size``.  A failed query returns 0, which selects the narrow
    payload for every shape -- always correct, just not specialised.
    """
    idx = int(device_index)
    if idx in _L2_BYTES:
        return _L2_BYTES[idx]

    value = 0
    try:
        props = torch.npu.get_device_properties(idx)
        value = int(getattr(props, "L2_cache_size", 0) or 0)
    except Exception:
        value = 0

    _L2_BYTES[idx] = value
    return value


def _pick_block(n, itemsize, l2_bytes):
    """Tile width in elements for ``n`` elements of ``itemsize`` bytes.

    ``l2_bytes`` is the live device L2 capacity (0 when unknown).  The in-place
    operator reads ``A``, reads ``B`` and writes ``A``, so its streaming footprint
    is ``3 * n * itemsize`` bytes; the wide payload is chosen only when that whole
    footprint fits the cache, i.e. only when the staged copies do not have to wait
    on DRAM.  Dispatch is deterministic on ``(n, itemsize, l2)`` by design -- see
    the module docstring on why an autotuner cannot be used here.
    """
    payload = _PAYLOAD_NARROW
    if l2_bytes and 3 * n * itemsize <= l2_bytes:
        payload = _PAYLOAD_WIDE

    block = payload // itemsize
    if block < _BLOCK_MIN:
        block = _BLOCK_MIN
    if block > _BLOCK_MAX:
        block = _BLOCK_MAX
    return block


def _grid_for(ntile):
    """Program count for ``ntile`` tiles, capped by the live vector-core count.

    The tiles a program cannot reach in one pass are picked up by the kernel's
    interleaved strided loop, so capping the grid is safe at any tensor size; the
    cap only exists to keep the 2**30-element cases inside the 1-D grid extent
    and to avoid oversubscribing the cores.
    """
    cap = _num_vector_cores() * _PROGRAMS_PER_CORE or _MAX_GRID_1D
    return (ntile if ntile < cap else cap,)


@triton.jit
def _bitwise_xor_inplace_kernel(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """``o[i] = a[i] ^ b[i]`` for ``n`` elements, ``n`` a multiple of ``BLOCK``.

    For the in-place operator ``o_ptr`` *is* ``a_ptr``: each tile is read into UB
    before anything is written back, and every program owns a disjoint set of
    tiles, so the update is race-free without any extra synchronisation.

    The body is deliberately branch-free and mask-free, so the compiler's
    automatic double buffering applies to exactly the three buffers the loop
    really uses (see the module docstring); the caller guarantees
    ``n % BLOCK == 0`` and the ragged remainder is handled by
    ``_bitwise_xor_masked_kernel``.

    Tiles are assigned to programs in an interleaved stride (``pid``,
    ``pid + nprog``, ...) so that the tiles being streamed at any instant lie in
    one contiguous region of the tensor.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntile = tl.cdiv(n, BLOCK)

    # Allocated once, outside the loop: a ``tile.alloc`` must not sit inside a
    # conditional region, and one staging pair per program is reused by every
    # tile the program owns.
    a_ub = tle.dsa.alloc([BLOCK], a_ptr.dtype.element_ty, tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc([BLOCK], b_ptr.dtype.element_ty, tle.dsa.ascend.UB)

    for tile in range(pid, ntile, nprog):
        base = tile * BLOCK
        a_bp = tl.make_block_ptr(
            a_ptr,
            shape=(n,),
            strides=(1,),
            offsets=(base,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        b_bp = tl.make_block_ptr(
            b_ptr,
            shape=(n,),
            strides=(1,),
            offsets=(base,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        tle.dsa.copy(a_bp, a_ub, [BLOCK])
        tle.dsa.copy(b_bp, b_ub, [BLOCK])
        tl.store(
            o_ptr + base + tl.arange(0, BLOCK),
            tle.dsa.to_tensor(a_ub) ^ tle.dsa.to_tensor(b_ub),
        )


@triton.jit
def _bitwise_xor_masked_kernel(a_ptr, b_ptr, o_ptr, start, n, BLOCK: tl.constexpr):
    """Masked ``o[i] = a[i] ^ b[i]`` over ``[start, n)``.

    Handles both the ragged remainder of a large tensor and, on its own, the
    whole of a small one.  The ``_bitwise_xor_inplace_kernel``'s reason to
    exclude this work is UB budget, not correctness: ``tle.dsa.copy`` has no
    mask, so an unmasked tile cannot be issued near the end of a tensor.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntile = tl.cdiv(n - start, BLOCK)

    for tile in range(pid, ntile, nprog):
        offs = start + tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(a_ptr + offs, mask=mask, other=0)
        y = tl.load(b_ptr + offs, mask=mask, other=0)
        tl.store(o_ptr + offs, x ^ y, mask=mask)


def _payload(t):
    """Zero-copy byte view for ``torch.bool``.

    ``bool`` occupies one byte per element, so the 8-bit payload view preserves
    the bit patterns -- and therefore the XOR result -- while giving the UB
    buffers and the vector unit a plain 8-bit integer type to work on.
    """
    if t.dtype == torch.bool:
        return t.view(torch.uint8)
    return t


_SUPPORTED_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def bitwise_xor_tensor_(A, B, out0=None, *args, **kwargs):
    """``A ^= B`` in place; returns the mutated tensor."""
    logger.debug("GEMS BITWISE_XOR_TENSOR_")
    if A.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.bitwise_xor import bitwise_xor_tensor_ as _generic

        return _generic(A, B)

    """``A ^= B`` in place (or ``out0 = A ^ B`` when ``out0`` is supplied).

    Returns the mutated tensor object itself, so the declared reference call form
    ``bitwise_xor_tensor_(A, B)`` satisfies ``out is A``.

    Only layout and metadata work happens on the host: the flattened views are
    zero-copy for dense inputs, and every numeric element is produced by the
    device kernels.  When it can be expressed the fused form is always valid
    (``out0 is A`` makes ``o_ptr`` and ``a_ptr`` the same allocation), so the
    split into a full-tile launch plus a masked remainder launch is purely a
    kernel-shape decision, never a semantic one.
    """
    target = A if out0 is None else out0

    if A.numel() == 0:
        return target

    a_flat = _payload(A).reshape(-1)
    b_flat = _payload(B).reshape(-1)
    o_flat = _payload(target).reshape(-1)
    n = a_flat.numel()

    if n <= _SMALL_LIMIT:
        _bitwise_xor_masked_kernel[_grid_for(triton.cdiv(n, _BLOCK_SMALL))](
            a_flat, b_flat, o_flat, 0, n, BLOCK=_BLOCK_SMALL
        )
        return target

    device_index = a_flat.device.index
    block = _pick_block(
        n, a_flat.element_size(), _l2_bytes(0 if device_index is None else device_index)
    )
    n_full = (n // block) * block

    if n_full:
        _bitwise_xor_inplace_kernel[_grid_for(n_full // block)](
            a_flat, b_flat, o_flat, n_full, BLOCK=block
        )
    if n_full < n:
        _bitwise_xor_masked_kernel[_grid_for(triton.cdiv(n - n_full, _BLOCK_SMALL))](
            a_flat, b_flat, o_flat, n_full, n, BLOCK=_BLOCK_SMALL
        )
    return target
