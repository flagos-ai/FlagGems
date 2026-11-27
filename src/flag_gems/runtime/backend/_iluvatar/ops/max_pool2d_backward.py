"""Triton implementation of max_pool2d_with_indices backward.

Semantics (matches ATen ``max_pool2d_with_indices_backward``): the forward pass
recorded, for every output position ``(n, c, oh, ow)``, the flat offset of the
arg-max inside the *unpadded* input plane, i.e. ``idx = h * W + w``.  Backward is
a pure scatter-add of ``grad_output`` onto those recorded offsets, with duplicate
windows accumulating.  ``input`` is used only for shape/dtype metadata - its
elements are never read.

Precision contract (measured, not assumed): the reference accumulates duplicate
contributions in **float32** and rounds to the output dtype exactly once
(verified: ATen is bitwise equal to a float32 scatter-add rounded once).  A
reduced-precision ``atomic_add`` rounds at every partial sum and drifts up to
9.7 absolute past the pytest tolerance (0.0001 atol / 0.001 rtol for fp16).
A per-input-element gather is also wrong for this operator: with stride 2 there
are only ``Ho*Wo = H*W/4`` output positions, so gathering touches 4x the
elements and measured 4.4x slower than the scatter.  Hence:

* fp32 output  -> the accumulation dtype already *is* the output dtype: zeroed
  output plus an ``atomic_add`` scatter (one pass).
* fp16/bf16    -> scatter into a float32 staging buffer, then round once with an
  elementwise cast kernel.

The scatter is the dominant pass and Profile measured it as *integer-ALU* bound
(integer ALU/BFU pipe at 96.4%, ~47 integer issues per window) rather than
DRAM bound, so three integer-work reductions are applied here:

1. ``N_OUT`` is a ``tl.constexpr``, turning the per-element runtime division
   ``offs // n_out`` into a multiply-shift strength reduction.
2. The target offset is computed in int32 (``WIDE=False``) with a host-side
   range guard that selects the int64 variant for tensors that need it.
3. The ``grad_output`` value is converted to float32 in a register *before*
   ``tl.atomic_add`` instead of inside it; the inline f16->f32 conversion made
   the compiler reserve 8 KiB/block of dynamic shared memory, capping
   theoretical occupancy at 50%.

Every grid is flat: the measured workloads reach ``N*C == 65536`` planes, so a
plane index must never occupy grid dimension y (device limit 65535).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT32_MAX = 2**31 - 1

# Plane-chunk budget for keeping the float32 staging buffer resident in the
# device's 16 MB L2 between its scatter and its cast.  Sized so that a chunk plus
# the streamed idx/grad_output traffic stays comfortably under the L2 capacity.
_L2_CHUNK_BYTES = 8 * 1024 * 1024


@triton.jit
def _mp2d_backward_scatter(
    go_ptr,
    idx_ptr,
    out_ptr,
    n_total,
    n_in,
    N_OUT: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total

    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    # explicit register conversion: keeps the f16->f32 cast out of the atomic
    g = tl.load(go_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    plane = offs // N_OUT
    if WIDE:
        valid = mask & (idx >= 0) & (idx < n_in)
        tgt = plane.to(tl.int64) * n_in + idx
    else:
        idx32 = idx.to(tl.int32)
        valid = mask & (idx32 >= 0) & (idx32 < n_in)
        tgt = plane * n_in + idx32
    tl.atomic_add(out_ptr + tgt, g, mask=valid, sem="relaxed")


@triton.jit
def _cast_dtype(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=mask)


def _pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def max_pool2d_backward(
    grad_output,
    input,
    indices,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    logger.debug("GEMS_ILUVATAR MAX_POOL2D_BACKWARD")
    # Geometry is *not* re-derived: Ho/Wo come from the recorded output shape so
    # that ceil_mode and every padding corner case is automatically exact.
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(kernel_size) if stride is None else _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)

    n_in = int(input.shape[-2]) * int(input.shape[-1])
    n_out = int(grad_output.shape[-2]) * int(grad_output.shape[-1])

    go = grad_output.contiguous()
    idx = indices.contiguous()
    n_out_total = int(go.numel())
    wide = (int(input.numel()) > _INT32_MAX) or (n_out_total > _INT32_MAX)

    # The scatter's 8 KiB/block dynamic-shared scratch (present only when the
    # grad_output value needs an f16->f32 conversion for the atomic) scales with
    # the per-thread element count, so BLOCK=512 at num_warps=4 keeps 4
    # elements/thread while halving the scratch: 4 KiB/block lets 32 blocks
    # reside per CU (= 128 warps) instead of 16 blocks (= 64 warps), lifting
    # theoretical occupancy from 50% to 100%.  The cast kernel already runs at
    # 100% occupancy and the streaming floor, so it keeps BLOCK=1024.
    SCATTER_BLOCK = 512
    CAST_BLOCK = 1024

    if input.dtype == torch.float32:
        out = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
        _mp2d_backward_scatter[(triton.cdiv(n_out_total, SCATTER_BLOCK),)](
            go,
            idx,
            out,
            n_out_total,
            n_in,
            N_OUT=n_out,
            WIDE=wide,
            BLOCK=SCATTER_BLOCK,
            num_warps=4,
        )
        return out

    # Reduced precision: the float32 staging data is 4*N*C*H*W bytes, which for
    # these workloads is ~12.9 MB against a 16 MB L2.  A single full-size staging
    # buffer costs two extra full-size DRAM passes: the torch.zeros fill is
    # evicted before the corresponding scatter reaches it, and the full-size cast
    # re-reads it mostly from DRAM (Profile measured a 1.36% L2 hit).  Splitting
    # the planes into chunks and doing fill -> scatter -> cast per chunk keeps
    # each chunk resident across its whole lifetime, so the staging writes,
    # atomic read-modify-writes and cast reads are all served by L2.  The
    # arithmetic is untouched: fp32 accumulation, exactly one rounding.
    n_total_in = int(input.numel())
    planes = n_total_in // n_in
    n_chunks = max(1, -(-(n_total_in * 4) // _L2_CHUNK_BYTES))
    out = torch.empty(input.shape, dtype=input.dtype, device=input.device)

    if n_chunks == 1:
        chunk = torch.zeros(n_total_in, dtype=torch.float32, device=input.device)
        _mp2d_backward_scatter[(triton.cdiv(n_out_total, SCATTER_BLOCK),)](
            go,
            idx,
            chunk,
            n_out_total,
            n_in,
            N_OUT=n_out,
            WIDE=wide,
            BLOCK=SCATTER_BLOCK,
            num_warps=4,
        )
        _cast_dtype[(triton.cdiv(n_total_in, CAST_BLOCK),)](
            chunk,
            out,
            n_total_in,
            BLOCK=CAST_BLOCK,
            num_warps=4,
        )
        return out

    per_chunk = -(-planes // n_chunks)
    out_flat = out.view(planes, n_in)
    go_flat = go.view(planes, n_out)
    idx_flat = idx.view(planes, n_out)

    for p0 in range(0, planes, per_chunk):
        p1 = min(p0 + per_chunk, planes)
        m = p1 - p0
        chunk_in = m * n_in
        chunk_out = m * n_out
        chunk = torch.zeros(chunk_in, dtype=torch.float32, device=input.device)
        _mp2d_backward_scatter[(triton.cdiv(chunk_out, SCATTER_BLOCK),)](
            go_flat[p0:p1],
            idx_flat[p0:p1],
            chunk,
            chunk_out,
            n_in,
            N_OUT=n_out,
            WIDE=wide,
            BLOCK=SCATTER_BLOCK,
            num_warps=4,
        )
        _cast_dtype[(triton.cdiv(chunk_in, CAST_BLOCK),)](
            chunk,
            out_flat[p0:p1],
            chunk_in,
            BLOCK=CAST_BLOCK,
            num_warps=4,
        )
    return out
