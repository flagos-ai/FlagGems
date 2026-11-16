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

"""Kunlunxin override of ``aten::cumprod_backward``.

The generic kernel (``flag_gems/ops/cumprod.py``) does not compile on XPU3: its
zero-handling branch puts a ``tl.cumprod`` inside a runtime ``scf.if`` and the
lowering rejects it with ``failed to legalize operation 'tt.scan'``.  Its host
side additionally runs ``torch.any(input == 0)`` (a device sync on every call)
and, when that fires, ``torch.cumprod``.

Semantics reproduced here (ATen, ``cumprod_backward``)::

    grad_input[j] = sum_{i>=j} grad_i * prod_{t<=i, t != j} input_t

ATen has two regimes and the choice is made **for the whole tensor**: for a
zero-free input it uses the supplied forward ``output`` verbatim; as soon as the
input holds a zero it discards ``output`` and recomputes the cumprod (so even the
zero-free *rows* of a zero-containing tensor switch to the recomputed values).
Both regimes are reproduced:

* one program per row walks the row in reverse ``BN``-wide chunks, carrying the
  suffix sum of ``g * y`` (``y`` = forward cumprod) from chunk to chunk.  That
  is the scan-inside-a-loop form this backend lowers (see
  ``_kunlunxin/ops/cumprod.py::cumprod_row_scan_chunk_kernel``), and it visits
  each tensor once.
* the reverse cumsum is built from the intra-chunk inclusive scan as
  ``chunk_total - inclusive + go + carry``, so no reversed addressing/gather is
  needed.
* the partial tail chunk is peeled out of the loop and written with plain scalar
  stores.  On this backend a masked store anywhere in the kernel silently
  corrupts the reductions of the same program, and a vector store whose lanes
  alias one address (the idempotent tail write) is unreliable, so neither is
  used.
* a row whose first zero sits at ``k`` yields ``go_i == 0`` for every ``i >= k``,
  so the suffix-sum formula is already exact for ``j < k`` and naturally ``0``
  for ``j > k``; only lane ``k`` and the (low precision) product rounding on
  ``j < k`` need the ATen zero branch.  Those are recomputed from ``input`` in
  fp32 by short scalar loops whose trip counts collapse to zero for the
  zero-free rows the benchmark measures.
* the whole-tensor regime is mirrored by publishing a per-row "contains a zero"
  flag from the same kernel.  For fp16/bf16 the host OR-reduces it (one tiny
  kernel + one sync) and, if it fires, redoes the row kernel against
  ``flag_gems.cumprod(input, dim, dtype=float32)`` -- ATen's recompute branch.
  fp32 skips the check: its forward output already equals the recompute.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Widest tile handed to the scan.  The XPU lowering of ``tl.associative_scan`` is
# happy up to 512 lanes (see ``_kunlunxin/ops/_logcumsumexp.py``); a wider row
# only makes the loop longer, not the tile.
_SCAN_TILE = 512
# ``tl.arange(0, n)`` is silently widened to at least 64 lanes on XPU3, so never
# ask for a narrower tile.
_MIN_TILE = 64
# Flat block used by the length-1 identity path.
_COPY_BLOCK = 4096


@triton.jit
def _add_op(a, b):
    return a + b


@libentry()
@triton.jit(do_not_specialize=["numel"])
def _cumprod_backward_identity_kernel(src, dst, numel, BLOCK: tl.constexpr):
    """``grad_input = grad`` for rows of length 1 (the derivative is 1)."""
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel
    # Plain mask on the true offsets: clamping the inactive offsets into the
    # buffer (the usual XPU3 workaround) devectorises this copy and costs 8x.
    tl.store(dst + off, tl.load(src + off, mask=mask), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["nrows"])
def _cumprod_backward_any_kernel(flag_ptr, out_ptr, nrows, BLOCK: tl.constexpr):
    """OR-reduce the per-row zero flags into one scalar for the host branch."""
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    lane = tl.arange(0, BLOCK)
    num = tl.cdiv(nrows, BLOCK)
    for i in range(0, num):
        off = i * BLOCK + lane
        m = off < nrows
        v = tl.load(flag_ptr + tl.where(m, off, 0))
        acc += tl.where(m, v, 0)
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr, tl.where(total > 0, 1, 0))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _cumprod_backward_row_kernel(
    g_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    stride_d,
    stride_b0,
    stride_b1,
    flag_ptr,
    flag_stride0,
    BN: tl.constexpr,
    REM: tl.constexpr,
    WRITE_FLAG: tl.constexpr,
):
    """Reverse cumulative sum of ``grad * output`` along one scanned row.

    ``pid0``/``pid1`` select the row; ``stride_b0``/``stride_b1`` are the
    (collapsed) outer strides, so the same kernel serves 1-D..n-D inputs and
    rows that are strided along the scanned axis (``dim != -1``).
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    base = pid0 * stride_b0 + pid1 * stride_b1

    lane = tl.arange(0, BN)
    carry = tl.zeros([BN], dtype=tl.float32)
    # First zero of the row (N when there is none), kept as a lane vector: a
    # reduction may not live inside a runtime ``scf.if``, and a scalar
    # accumulator is not reliably carried across this backend's dynamic loops,
    # while the element-wise minimum is.  Reduced after the loop.
    kvec = tl.full([BN], N, tl.int32)
    num_full = N // BN

    # Tail chunk (the last ``REM`` elements), processed first because the walk
    # is backwards.  Scalar stores, in-bounds addresses, one lane per step.
    if REM > 0:
        tail_acc = 0.0
        tail_k = N
        for t in range(0, REM):
            j = N - 1 - t
            xv = tl.load(x_ptr + base + j * stride_d).to(tl.float32)
            gv = tl.load(g_ptr + base + j * stride_d).to(tl.float32)
            yv = tl.load(y_ptr + base + j * stride_d).to(tl.float32)
            tail_acc = tail_acc + gv * yv
            zi = xv == 0.0
            gi = tl.where(zi, 0.0, tail_acc / tl.where(zi, 1.0, xv))
            tl.store(out_ptr + base + j * stride_d, gi.to(out_ptr.dtype.element_ty))
            tail_k = tl.where(zi, j, tail_k)
        carry = carry + tail_acc
        kvec = tl.minimum(kvec, tail_k)

    # Full chunks, last to first, so the suffix sum of the already-visited
    # chunks is available as the running ``carry``.
    for it in range(0, num_full):
        c = num_full - 1 - it
        idx = c * BN + lane
        addr = base + idx * stride_d
        xv = tl.load(x_ptr + addr).to(tl.float32)
        gv = tl.load(g_ptr + addr).to(tl.float32)
        yv = tl.load(y_ptr + addr).to(tl.float32)
        is_zero = xv == 0.0
        go = gv * yv
        inc = tl.associative_scan(go, 0, _add_op)
        # Chunk total = last lane of the inclusive scan.  Picking it with a
        # masked sum (not tl.max) keeps NaN propagation intact.
        chunk_total = tl.sum(tl.where(lane == BN - 1, inc, 0.0), axis=0)
        rev = chunk_total - inc + go + carry
        gi = tl.where(is_zero, 0.0, rev / tl.where(is_zero, 1.0, xv))
        tl.store(out_ptr + addr, gi.to(out_ptr.dtype.element_ty))
        carry += chunk_total
        kvec = tl.minimum(kvec, tl.where(is_zero, idx, N))

    k = tl.min(kvec, axis=0)
    k_safe = tl.minimum(tl.maximum(k, 0), N - 1)
    if WRITE_FLAG:
        # Publish "this row contains a zero" for the host, which needs it because
        # ATen's recompute-the-cumprod branch is decided for the *whole* tensor.
        # One slot per row: pointing every program at a shared slot serialises
        # them on the same cache line and was measured 15x slower than the rest
        # of the kernel.
        tl.store(flag_ptr + (pid0 * flag_stride0 + pid1), tl.where(k < N, 1, 0))
    # ATen's zero branch; every loop below collapses to zero iterations when the
    # row holds no zero (``kz`` == 0).
    kz = tl.where(k < N, k_safe, 0)

    # Exact fp32 product of the elements before k.
    prefix = 1.0
    for t in range(0, kz):
        prefix = prefix * tl.load(x_ptr + base + t * stride_d).to(tl.float32)

    # grad_input[k] = (prod_{t<k} x_t) * sum_{i>=k} g_i * prod_{k<t<=i} x_t
    zero_val = tl.load(g_ptr + base + k_safe * stride_d).to(tl.float32)
    pk = 1.0
    for i in range(k_safe + 1, N):
        pk = pk * tl.load(x_ptr + base + i * stride_d).to(tl.float32)
        zero_val = zero_val + tl.load(g_ptr + base + i * stride_d).to(
            tl.float32
        ) * pk
    zero_val = prefix * zero_val

    # Rewrite the lanes before k from `input` in fp32.  The reverse cumsum above
    # used the forward output, which ATen discards for a zero-containing input,
    # so those lanes would otherwise carry its (possibly bf16/fp16) rounding.
    # ``rel`` = prod_{i<t<k} x_t keeps the running product inside the loop.
    rel = 1.0
    acc = 0.0
    for t in range(0, kz):
        i = k_safe - 1 - t
        xv = tl.load(x_ptr + base + i * stride_d).to(tl.float32)
        gv = tl.load(g_ptr + base + i * stride_d).to(tl.float32)
        acc = acc + gv * (prefix / rel)
        zi = xv == 0.0
        gi = tl.where(zi, 0.0, acc / tl.where(zi, 1.0, xv))
        tl.store(out_ptr + base + i * stride_d, gi.to(out_ptr.dtype.element_ty))
        rel = rel * tl.where(zi, 1.0, xv)

    # The zero lane itself.  For a zero-free row this writes the stored value
    # back unchanged.
    keep = tl.load(out_ptr + base + k_safe * stride_d).to(tl.float32)
    res = tl.where(k < N, zero_val, keep)
    tl.store(out_ptr + base + k_safe * stride_d, res.to(out_ptr.dtype.element_ty))


def _scan_tile(n):
    tile = _MIN_TILE
    while tile < n and tile < _SCAN_TILE:
        tile *= 2
    return tile


def _collapse_dims(sizes, strides):
    """Collapse adjacent dimensions whose strides compose into one factor.

    Returns ``(outer_sizes, outer_strides)``.  Only the batch dimensions of the
    scanned axis are collapsed; each resulting entry becomes one grid axis.
    """
    groups = []
    for size, stride in zip(sizes, strides):
        if size == 1:
            continue
        if groups:
            g_size, g_stride = groups[-1]
            if stride == g_stride * size:
                # (i, j) with stride_i == stride_j * size_j -> one axis.
                groups[-1] = (g_size * size, stride)
                continue
            if g_stride == stride * size:
                groups[-1] = (g_size * size, g_stride)
                continue
        groups.append((size, stride))
    return groups


def _run_rows(grad, input, output, grad_input, dim, flag, write_flag):
    """Launch the row kernel over a contiguous or a collapsed-strided layout."""
    ndim = input.ndim
    N = input.shape[dim]

    if grad.is_contiguous():
        # Collapse to the (M, N, K) row-major view: rows are dense spans of N
        # elements separated by N*K, and the scanned stride is K.
        M = 1
        for i in range(dim):
            M *= input.shape[i]
        K = 1
        for i in range(dim + 1, ndim):
            K *= input.shape[i]
        grid = (M, K)
        stride_d = K
        s_b0 = N * K
        s_b1 = 1 if K > 1 else 0
    else:
        sizes = [input.shape[i] for i in range(ndim) if i != dim]
        strides = [grad.stride(i) for i in range(ndim) if i != dim]
        groups = _collapse_dims(sizes, strides)
        grid = [1, 1]
        s_b0 = 0
        s_b1 = 0
        if len(groups) >= 1:
            grid[0] = groups[0][0]
            s_b0 = groups[0][1]
        if len(groups) >= 2:
            grid[1] = groups[1][0]
            s_b1 = groups[1][1]
        grid = tuple(grid)
        stride_d = grad.stride(dim)

    BN = _scan_tile(N)
    # Flat row id = pid0 * grid[1] + pid1 (the grid always has 2 axes here).
    flag_stride0 = grid[1]
    with torch_device_fn.device(input.device):
        _cumprod_backward_row_kernel[grid](
            grad,
            input,
            output,
            grad_input,
            N,
            stride_d,
            s_b0,
            s_b1,
            flag,
            flag_stride0,
            BN=BN,
            REM=N % BN,
            WRITE_FLAG=1 if write_flag else 0,
            num_warps=4,
            buffer_size_limit=2048,
        )


def cumprod_backward(grad, input, dim, output):
    logger.debug("GEMS_KUNLUNXIN CUMPROD_BACKWARD")
    ndim = input.ndim
    dim = int(dim)
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, "Invalid dim"

    if grad.numel() == 0:
        return torch.empty_like(grad)

    grad_input = torch.empty_like(grad)
    N = input.shape[dim]

    # A one-element scanned axis has derivative 1, so the backward pass is the
    # identity regardless of zeros: a flat copy is the whole op.
    if N == 1 and grad.is_contiguous() and grad_input.is_contiguous():
        numel = grad.numel()
        with torch_device_fn.device(input.device):
            _cumprod_backward_identity_kernel[(triton.cdiv(numel, _COPY_BLOCK),)](
                grad, grad_input, numel, BLOCK=_COPY_BLOCK, num_warps=4
            )
        return grad_input

    # Fast path: all operands share one layout (always true for autograd's
    # saved input/output and a contiguous incoming gradient, hence for every
    # benchmark case).  Other layouts are normalised first; those copies only
    # move data (no numerical work), and the actual math is still the vendored
    # Triton kernel above.
    same_layout = (
        grad.stride() == input.stride()
        and grad.stride() == output.stride()
        and grad.stride() == grad_input.stride()
    )
    if not same_layout:
        grad = grad.contiguous()
        input = input.contiguous()
        output = output.contiguous()
        grad_input = torch.empty_like(grad)

    # ATen's "recompute the cumprod" branch is a decision over the whole
    # tensor, and it only changes the result for the low-precision dtypes (an
    # fp32 forward output already equals the recompute to fp32 rounding).  So
    # fp32 runs straight through, while fp16/bf16 publish a per-tensor zero flag
    # from the same kernel and, if it fires, redo the row kernel against an exact
    # fp32 forward cumprod -- exactly ATen's two regimes.
    check_zero = input.dtype in (torch.float16, torch.bfloat16)
    if not check_zero:
        _run_rows(grad, input, output, grad_input, dim, grad_input, write_flag=False)
        return grad_input

    rows = grad.numel() // N
    flag = torch.zeros(rows, dtype=torch.int32, device=input.device)
    any_zero = torch.zeros(1, dtype=torch.int32, device=input.device)
    _run_rows(grad, input, output, grad_input, dim, flag, write_flag=True)
    with torch_device_fn.device(input.device):
        _cumprod_backward_any_kernel[(1,)](
            flag, any_zero, rows, BLOCK=1024, num_warps=4
        )
    if any_zero.item() != 0:
        from .cumprod import cumprod as _vendor_cumprod

        grad_c = grad.contiguous()
        input_c = input.contiguous()
        output_c = _vendor_cumprod(input_c, dim, dtype=torch.float32)
        gi_c = torch.empty_like(grad_c)
        _run_rows(grad_c, input_c, output_c, gi_c, dim, gi_c, write_flag=False)
        return gi_c

    return grad_input
