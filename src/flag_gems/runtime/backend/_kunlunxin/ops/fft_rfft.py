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

"""Kunlunxin XPU3 override for ``aten::fft_rfft``.

The generic KernelGen kernel is a tiled direct DFT that accumulates
``acc_re += tl.sum(x_row[None, :] * cos(2*pi*(j*k mod n)/n), axis=1)`` inside a
dynamic ``for j0`` loop.  On XPU3 that is the known *loop-carried reduce*
encoding trap: the ``tt.reduce`` result carries a different encoding than the
``tl.arange``-based 1D accumulator, and the whole op fails to compile::

    RuntimeError: PassManager::run failed: loc("acc_re"): 'arith.addf' op
    requires the same encoding for all operands and results
    -> triton.runtime.errors.OutOfResources: out of resource: uni_sram

so **every** kernel-invoking test cell fails (baseline on this backend: 29
failed / 14 passed / 3 skipped, the passes are the argument-validation cells).
Hoisting the reduction out of the loop or accumulating a 2-D tile also fails
(``'arith.addi' encoding`` / ``uni_sram`` in ``TritonXPUUnrollControl``), see
``harness/perf_ir/fft_rfft/probe_variants*.log``.

The fix removes ``tt.reduce`` entirely: the DFT is written as a *no-reduction*
kernel where one lane owns one output bin (``k``) and a runtime loop walks the
input samples ``j``, accumulating ``x_j * cos/sin(theta_jk)`` as a
scalar-vector broadcast.  ``BLOCK_K`` is capped at 64 because that is the only
1-D tile width this backend lowers reliably for a multi-vector accumulator
(128/256/512/1024 all blow up in ``TritonXPUUnrollControl``; measured).

The cost model changes from ``O(n * K / BLOCK_K)`` vector lanes to
``n`` scalar loads plus ``n * K`` MACs per program, which is irrelevant at the
sizes this direct-DFT fallback covers and keeps the op compiling at all.
Everything else (schema, errors, empty tensors, dtype promotion, ``dim``
handling, non-contiguous inputs, ``norm``) mirrors the generic operator
exactly; there is no aten/CPU fallback of any kind.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["n"])
def _rfft_lastdim_kernel(
    x_ptr,
    out_ptr,
    N,
    n,
    K,
    BLOCK_K: tl.constexpr,
    IS_F64: tl.constexpr,
    NORM: tl.constexpr,
):
    # One program per (batch row, BLOCK_K output bins).  No tt.reduce: every
    # lane accumulates its own bin while the loop streams one input sample at
    # a time from global memory.
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BLOCK_K)
    r = pid // nk
    kb = pid - r * nk

    off_in = r * N
    off_out = r * K

    k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = k < K
    keff = tl.where(kmask, k, 0)

    if IS_F64:
        acc_re = tl.zeros((BLOCK_K,), dtype=tl.float64)
        acc_im = tl.zeros((BLOCK_K,), dtype=tl.float64)
        step = 6.283185307179586476925286766559 / n.to(tl.float64)
    else:
        acc_re = tl.zeros((BLOCK_K,), dtype=tl.float32)
        acc_im = tl.zeros((BLOCK_K,), dtype=tl.float32)
        step = 6.283185307179586476925286766559 / n.to(tl.float32)

    # ``n > N`` zero-pads: only the first min(n, N) samples carry data.
    nj = tl.minimum(n, N)

    for j in range(0, nj):
        xj = tl.load(x_ptr + (off_in + j))
        if IS_F64:
            xv = xj.to(tl.float64)
        else:
            xv = xj.to(tl.float32)
        m = (j * keff) % n
        if IS_F64:
            ang = m.to(tl.float64) * step
        else:
            ang = m.to(tl.float32) * step
        acc_re += xv * tl.cos(ang)
        acc_im -= xv * tl.sin(ang)

    # DC (k == 0) and the even-length Nyquist bin are real for real input.
    is_nyq = (keff * 2) == n
    im0 = (keff == 0) | ((n % 2 == 0) & is_nyq)
    acc_im = tl.where(im0, acc_im * 0.0, acc_im)

    if NORM == 1:
        if IS_F64:
            scale = 1.0 / n.to(tl.float64)
        else:
            scale = 1.0 / n.to(tl.float32)
        acc_re = acc_re * scale
        acc_im = acc_im * scale
    elif NORM == 2:
        if IS_F64:
            scale = 1.0 / tl.sqrt(n.to(tl.float64))
        else:
            scale = 1.0 / tl.sqrt(n.to(tl.float32))
        acc_re = acc_re * scale
        acc_im = acc_im * scale

    tmp = (off_out + k) * 2
    tl.store(out_ptr + tmp, acc_re, mask=kmask)
    tl.store(out_ptr + tmp + 1, acc_im, mask=kmask)


@triton.jit(do_not_specialize=["n"])
def _rfft_dft_kernel(
    x_ptr,
    out_ptr,
    N,
    n,
    K,
    sd,
    od,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    q0,
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    BLOCK_K: tl.constexpr,
    IS_F64: tl.constexpr,
    NORM: tl.constexpr,
):
    # General path: non-trailing ``dim`` and/or non-contiguous input.  Row
    # bases are rebuilt in-kernel from an explicit size/divisor/stride
    # decomposition of the flattened batch (8 padded slots), so no host-side
    # data movement is needed.
    pid = tl.program_id(0)
    nk = tl.cdiv(K, BLOCK_K)
    r = pid // nk
    kb = pid - r * nk

    d0 = r // q0
    d1 = r // q1
    d2 = r // q2
    d3 = r // q3
    d4 = r // q4
    d5 = r // q5
    d6 = r // q6
    d7 = r // q7

    off_in = (
        ((d0 % s0) * i0)
        + ((d1 % s1) * i1)
        + ((d2 % s2) * i2)
        + ((d3 % s3) * i3)
        + ((d4 % s4) * i4)
        + ((d5 % s5) * i5)
        + ((d6 % s6) * i6)
        + ((d7 % s7) * i7)
    )
    off_out = (
        ((d0 % s0) * o0)
        + ((d1 % s1) * o1)
        + ((d2 % s2) * o2)
        + ((d3 % s3) * o3)
        + ((d4 % s4) * o4)
        + ((d5 % s5) * o5)
        + ((d6 % s6) * o6)
        + ((d7 % s7) * o7)
    )

    k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = k < K
    keff = tl.where(kmask, k, 0)

    if IS_F64:
        acc_re = tl.zeros((BLOCK_K,), dtype=tl.float64)
        acc_im = tl.zeros((BLOCK_K,), dtype=tl.float64)
        step = 6.283185307179586476925286766559 / n.to(tl.float64)
    else:
        acc_re = tl.zeros((BLOCK_K,), dtype=tl.float32)
        acc_im = tl.zeros((BLOCK_K,), dtype=tl.float32)
        step = 6.283185307179586476925286766559 / n.to(tl.float32)

    nj = tl.minimum(n, N)

    for j in range(0, nj):
        xj = tl.load(x_ptr + (off_in + j * sd))
        if IS_F64:
            xv = xj.to(tl.float64)
        else:
            xv = xj.to(tl.float32)
        m = (j * keff) % n
        if IS_F64:
            ang = m.to(tl.float64) * step
        else:
            ang = m.to(tl.float32) * step
        acc_re += xv * tl.cos(ang)
        acc_im -= xv * tl.sin(ang)

    is_nyq = (keff * 2) == n
    im0 = (keff == 0) | ((n % 2 == 0) & is_nyq)
    acc_im = tl.where(im0, acc_im * 0.0, acc_im)

    if NORM == 1:
        if IS_F64:
            scale = 1.0 / n.to(tl.float64)
        else:
            scale = 1.0 / n.to(tl.float32)
        acc_re = acc_re * scale
        acc_im = acc_im * scale
    elif NORM == 2:
        if IS_F64:
            scale = 1.0 / tl.sqrt(n.to(tl.float64))
        else:
            scale = 1.0 / tl.sqrt(n.to(tl.float32))
        acc_re = acc_re * scale
        acc_im = acc_im * scale

    tmp = (off_out + k * od) * 2
    tl.store(out_ptr + tmp, acc_re, mask=kmask)
    tl.store(out_ptr + tmp + 1, acc_im, mask=kmask)


def _next_pow2(v):
    v = max(1, int(v))
    return 1 << (v - 1).bit_length()


def _pad8(lst, fill):
    lst = list(lst)[:8]
    return lst + [fill] * (8 - len(lst))


def fft_rfft(self, n=None, dim=-1, norm=None):
    # ``aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None)``
    logger.debug("GEMS_KUNLUNXIN FFT_RFFT")

    x = self
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.is_complex():
        raise RuntimeError(
            "rfft expects a real input tensor, but got %s"
            % ("ComplexDouble" if x.dtype == torch.complex128 else "ComplexFloat")
        )
    if norm is not None and norm not in ("backward", "forward", "ortho"):
        raise RuntimeError('Invalid normalization mode: "%s"' % norm)

    ndim = x.dim()
    d = int(dim)
    if d < 0:
        d += ndim
    if ndim == 0:
        raise IndexError(
            "Dimension specified as %d but tensor has no dimensions" % int(dim)
        )
    if d < 0 or d >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of [%d, %d], but got %d)"
            % (-ndim, ndim - 1, int(dim))
        )

    N = int(x.shape[d])
    nn = N if n is None else int(n)
    if nn <= 0:
        raise RuntimeError("Invalid number of data points (%d) specified" % nn)
    K = nn // 2 + 1

    shape_out = list(x.shape)
    shape_out[d] = K

    is_f64 = x.dtype == torch.float64
    cplx = torch.complex128 if is_f64 else torch.complex64
    fdt = torch.float64 if is_f64 else torch.float32

    if x.numel() == 0 or K <= 0 or N <= 0 or nn <= 0:
        return torch.empty(shape_out, dtype=cplx, device=x.device)

    out_f = torch.empty(shape_out + [2], dtype=fdt, device=x.device)

    if norm is None or norm == "backward":
        ncode = 0
    elif norm == "forward":
        ncode = 1
    else:
        ncode = 2

    # BLOCK_K is the only free tile parameter.  XPU3 lowers a 1-D accumulator
    # of this shape reliably only up to 64 lanes (128/256/512/1024 die in
    # ``TritonXPUUnrollControl``), so cap it there; small transforms get one
    # k-block per row.
    blk_k = min(max(16, _next_pow2(K)), 64)
    nk = (K + blk_k - 1) // blk_k

    # Fast path: contiguous input, transform axis trailing -> row base r * N.
    if d == ndim - 1 and x.is_contiguous():
        R = x.numel() // N
        _rfft_lastdim_kernel[(R * nk,)](
            x,
            out_f,
            N,
            nn,
            K,
            BLOCK_K=blk_k,
            IS_F64=is_f64,
            NORM=ncode,
            num_warps=4,
        )
        return torch.view_as_complex(out_f)

    batch_sizes = [int(x.shape[i]) for i in range(ndim) if i != d]
    batch_istr = [int(x.stride(i)) for i in range(ndim) if i != d]

    ost = [1] * ndim
    acc = 1
    for i in range(ndim - 1, -1, -1):
        ost[i] = acc
        acc *= shape_out[i]
    batch_ostr = [ost[i] for i in range(ndim) if i != d]

    R = 1
    for s in batch_sizes:
        R *= s
    if R == 0:
        return torch.view_as_complex(out_f)

    # q[i] = product of the batch sizes at and after batch axis i, so the
    # divisor needed for axis i is q[i + 1] (1 for the innermost axis).
    m = len(batch_sizes)
    q = [1] * (m + 1)
    for i in range(m - 1, -1, -1):
        q[i] = q[i + 1] * batch_sizes[i]

    # Padded (size 1, divisor 1, stride 0) tail slots contribute nothing.
    s_pad = _pad8(batch_sizes, 1)
    q_pad = _pad8(q[1 : m + 1], 1)
    i_pad = _pad8(batch_istr, 0)
    o_pad = _pad8(batch_ostr, 0)

    _rfft_dft_kernel[(R * nk,)](
        x,
        out_f,
        N,
        nn,
        K,
        int(x.stride(d)),
        int(ost[d]),
        s_pad[0],
        s_pad[1],
        s_pad[2],
        s_pad[3],
        s_pad[4],
        s_pad[5],
        s_pad[6],
        s_pad[7],
        q_pad[0],
        q_pad[1],
        q_pad[2],
        q_pad[3],
        q_pad[4],
        q_pad[5],
        q_pad[6],
        q_pad[7],
        i_pad[0],
        i_pad[1],
        i_pad[2],
        i_pad[3],
        i_pad[4],
        i_pad[5],
        i_pad[6],
        i_pad[7],
        o_pad[0],
        o_pad[1],
        o_pad[2],
        o_pad[3],
        o_pad[4],
        o_pad[5],
        o_pad[6],
        o_pad[7],
        BLOCK_K=blk_k,
        IS_F64=is_f64,
        NORM=ncode,
        num_warps=4,
    )
    return torch.view_as_complex(out_f)


__all__ = ["fft_rfft"]
