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
"""Kunlunxin (XPU) linalg_householder_product.

The general implementation (``src/flag_gems/ops/linalg_householder_product.py``)
keeps ``Q`` as an ``[m, n]`` tile and reduces the reflector against it with
``tl.sum(v[:, None] * q_block, axis=0)`` inside a loop whose bound ``K`` is a
runtime value.  Both halves of that are unusable on TritonXPU:

* a 2-D ``axis=0`` reduction is not lowered ("axis must not be 0 for 2D+
  shapes, consider manually transpose"), and
* a dynamic loop wrapped around a 2-D reduction always exhausts ``uni_sram``.

so every single case of ``tests/test_linalg_householder_product.py`` dies with
``OutOfResources: uni_sram PassManager::run failed`` -- including the smallest
``(4, 3)`` one, i.e. it is a structural compile failure, not a tile-size issue.

This backend-local version keeps the same algorithm (``Q <- H(i) Q`` for
``i = k-1 .. 0`` starting from ``I[:, :n]``) but stores the accumulator
TRANSPOSED, ``W[c, r] = Q[r, c]``.  Because ``H(i)`` is symmetric,
``Q <- H(i) Q`` becomes ``W <- W H(i)``, i.e.

    W[c, :] -= tau_i * (W[c, :] . v_i) * v_i

so the reduction now runs along the LAST (contiguous) tile axis and is a plain
2-D ``axis=1`` ``tl.sum``.  The sweep itself is driven from the host, one pair
of launches per reflector, because a dynamic loop may not wrap the reduce.

For the padded row length 128 -- which covers the whole accuracy and benchmark
matrix (m <= 128) -- there is a second, much cheaper path: one program owns a
single output column, keeps it in registers and walks every reflector inside
the kernel.  That turns the reduction into a 1-D -> scalar ``tl.sum``, which a
dynamic loop MAY wrap, so the whole sweep becomes a single launch instead of
``2k`` of them (measured 4.8x - 22x faster on the benchmark shapes).  Its
compile envelope is not monotonic in the tile width, so only the validated
width is used; see ``_SWEEP_ROW``.

Backend rules this file obeys (all previously measured on this platform, see
harness/solution/performance/linalg_lstsq_xpu3_20260829.md):

* masked stores are NOT honoured (the whole tile is written) and every vector
  store touches exactly 64 contiguous elements whatever length was asked for
  => every buffer is padded so that all stores are unmasked, 64-element
  aligned and a multiple of 64 wide; the result is produced through an
  over-allocated flat buffer of which only the first ``numel`` elements are
  handed out.
* masked loads and ``other=`` are unreliable => all loads are unmasked, with
  out-of-range lanes clamped to a legal address and neutralised by the padding
  (zeros) they read.
* a 1-D vector broadcast into a tile that is then stored is silently wrong
  once the broadcast axis exceeds 128 elements => both rank-1 operands arrive
  as stride-0 duplicated-address 2-D loads.
* a square tile feeding a 2-D reduce exhausts uni_sram => the reduction tile is
  always 64 rows by >= 128 columns.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float32, torch.float64)

# Widest padded row a single reduction tile may span (validated on this
# platform for the 64 x MP shape used here).
_MAX_ROW = 8192

# Every vector store writes exactly 64 contiguous elements on this backend, so
# 64 is the granularity of every buffer row and of the c-tile.
_LANES = 64

# A square tile feeding a 2-D reduce OOMs uni_sram, and a 32-wide tile that is
# both read and written in one kernel comes back wrong.  The reduction axis is
# therefore padded to at least 128 while the c axis stays at 64.
_MIN_ROW = 128

# Padded row length for which the single-launch sweep below is validated.
#
# ``_sweep_kernel`` keeps one output column in registers and walks all
# reflectors inside the kernel, so it replaces ``2k`` launches with one; on the
# benchmark matrix that is 4.8x - 22x faster than the per-step path.  Its
# compile envelope is however NOT monotonic in the tile width: measured on this
# platform MP = 128, 1024 and 2048 build while MP = 256, 512 and 4096 all die
# with ``uni_sram PassManager::run failed`` inside ``TritonXPUUnrollControl``,
# and neither ``unroll_num`` nor ``buffer_size_limit`` nor ``num_warps`` moves
# that boundary.  Since 1024/2048 building is not something that can be
# extrapolated from, only the fully exercised MP = 128 is taken.
_SWEEP_ROW = 128


def _p2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


# ---------------------------------------------------------------------------
# W[c, r] = Q[r, c] initialised to the first N columns of the identity.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _init_w_kernel(
    W,
    N,
    M,
    WB,
    BC: tl.constexpr,
    MP: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, MP)
    keep = (c[:, None] < N) & (r[None, :] < M)
    val = tl.where(keep & (c[:, None] == r[None, :]), 1.0, 0.0)
    tl.store(W + b * WB + c[:, None] * MP + r[None, :], val)


# ---------------------------------------------------------------------------
# V[i, r] = v_i[r]  and  U[i, r] = tau_i * v_i[r].
#
# tau is folded in here so that the per-step kernels take no scalar argument at
# all: they receive V[:, i] / U[:, i] as pre-sliced tensors, which keeps the
# libentry cache to a single entry per shape instead of one per reflector.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _init_v_kernel(
    A,
    TAU,
    V,
    U,
    K,
    M,
    a_bs,
    a_rs,
    a_cs,
    t_bs,
    t_cs,
    VB,
    BI: tl.constexpr,
    MP: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1) * BI + tl.arange(0, BI)
    r = tl.arange(0, MP)
    ic = tl.minimum(i, K - 1)
    rc = tl.minimum(r, M - 1)
    a = tl.load(A + b * a_bs + rc[None, :] * a_rs + ic[:, None] * a_cs)
    t = tl.load(TAU + b * t_bs + ic[:, None] * t_cs + r[None, :] * 0)
    keep = (i[:, None] < K) & (r[None, :] < M)
    v = tl.where(
        r[None, :] > i[:, None], a, tl.where(r[None, :] == i[:, None], 1.0, 0.0)
    )
    v = tl.where(keep, v, 0.0)
    off = b * VB + i[:, None] * MP + r[None, :]
    tl.store(V + off, v)
    tl.store(U + off, v * t)


# ---------------------------------------------------------------------------
# S[c] = W[c, :] . v_i   -- the only reduction in the file, 2-D axis=1.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _dot_kernel(
    W,
    VI,
    S,
    WB,
    VB,
    SB,
    BC: tl.constexpr,
    MP: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, MP)
    # the operand that is contiguous along the reduction axis must be loaded
    # first and multiplied from the left, otherwise the reduce is wrong.
    t = tl.load(W + b * WB + c[:, None] * MP + r[None, :])
    vt = tl.load(VI + b * VB + r[None, :] + c[:, None] * 0)
    tl.store(S + b * SB + c, tl.sum(t * vt, axis=1))


# ---------------------------------------------------------------------------
# W[c, r] -= S[c] * (tau_i * v_i[r])
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _upd_kernel(
    W,
    S,
    UI,
    WB,
    SB,
    UB,
    BC: tl.constexpr,
    MP: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.arange(0, MP)
    off = b * WB + c[:, None] * MP + r[None, :]
    st = tl.load(S + b * SB + c[:, None] + r[None, :] * 0)
    ut = tl.load(UI + b * UB + r[None, :] + c[:, None] * 0)
    tl.store(W + off, tl.load(W + off) - st * ut)


# ---------------------------------------------------------------------------
# Whole sweep for one output column, kept in registers.
#
# One program owns column c of Q, i.e. row c of W, so the reduction is a plain
# 1-D -> scalar tl.sum and the reflector loop can live inside the kernel: the
# rule that forbids a dynamic loop around a reduction only bites for 2-D
# reductions.  Nothing is read back from W, so there is no read/write aliasing
# and no need to materialise the identity first.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _sweep_kernel(
    W,
    V,
    U,
    K,
    N,
    M,
    WB,
    VB,
    MP: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    r = tl.arange(0, MP)
    q = tl.where((c < N) & (r < M) & (r == c), 1.0, 0.0)
    for t in range(0, K):
        i = K - 1 - t
        v = tl.load(V + b * VB + i * MP + r)
        u = tl.load(U + b * VB + i * MP + r)
        q = q - tl.sum(q * v) * u
    tl.store(W + b * WB + c * MP + r, q)


# ---------------------------------------------------------------------------
# Transpose back into a flat, contiguous output.
#
# A transposing STORE would write its own values correctly and corrupt an
# unrelated allocation, so the transpose is done on the LOAD side: every
# program writes one 64-element contiguous chunk of the flat result and gathers
# the values it needs from W.
# ---------------------------------------------------------------------------
@libentry()
@triton.jit
def _out_kernel(
    OUT,
    W,
    TOTAL,
    MN,
    N,
    WB,
    MP,
    BT: tl.constexpr,
):
    f = tl.program_id(0) * BT + tl.arange(0, BT)
    fc = tl.minimum(f, TOTAL - 1)
    b = fc // MN
    rem = fc - b * MN
    r = rem // N
    c = rem - r * N
    tl.store(OUT + f, tl.load(W + b * WB + c * MP + r))


def linalg_householder_product(A, tau):
    """Q = H(0) H(1) ... H(k-1) restricted to its first n columns.

    ``H(i) = I - tau[i] v_i v_i^T`` with ``v_i[j] = 0`` for ``j < i``,
    ``v_i[i] = 1`` and ``v_i[j] = A[j, i]`` for ``j > i`` -- the geqrf layout.
    """
    logger.debug("GEMS LINALG_HOUSEHOLDER_PRODUCT")

    assert (
        A.dtype in _SUPPORTED_DTYPES
    ), f"linalg_householder_product only supports float32 and float64, got {A.dtype}"
    shape = A.shape
    if len(shape) < 2:
        raise ValueError("A must be at least 2D")

    m = shape[-2]
    n = shape[-1]
    k = tau.shape[-1]
    batch = 1
    for d in shape[:-2]:
        batch *= d

    dt, dev = A.dtype, A.device
    total = batch * m * n
    if total == 0:
        return torch.empty(shape, dtype=dt, device=dev)

    A3 = A.reshape(batch, m, n)
    tau2 = tau.reshape(batch, k)

    MP = max(_MIN_ROW, _p2(m))
    if MP > _MAX_ROW:
        raise NotImplementedError(
            "kunlunxin linalg_householder_product: the reduction tile spans the"
            f" padded row length {MP} > {_MAX_ROW}, which is outside the"
            " validated envelope of this backend"
        )
    NP = max(_LANES, _p2(n))
    KP = max(_LANES, _p2(k))
    nb = NP // _LANES
    BT = _LANES
    npad = triton.cdiv(total, BT) * BT

    W = torch.empty((batch, NP, MP), dtype=dt, device=dev)
    V = torch.empty((batch, KP, MP), dtype=dt, device=dev)
    U = torch.empty((batch, KP, MP), dtype=dt, device=dev)
    S = torch.empty((batch, NP), dtype=dt, device=dev)
    OUT = torch.empty((npad,), dtype=dt, device=dev)

    WB = NP * MP
    VB = KP * MP

    with torch_device_fn.device(dev):
        if k == 0:
            _init_w_kernel[(batch, nb)](W, n, m, WB, BC=_LANES, MP=MP)
        else:
            _init_v_kernel[(batch, KP // _LANES)](
                A3,
                tau2,
                V,
                U,
                k,
                m,
                A3.stride(0),
                A3.stride(1),
                A3.stride(2),
                tau2.stride(0),
                tau2.stride(1),
                VB,
                BI=_LANES,
                MP=MP,
            )
            if MP == _SWEEP_ROW:
                _sweep_kernel[(batch, n)](W, V, U, k, n, m, WB, VB, MP=MP)
            else:
                _init_w_kernel[(batch, nb)](W, n, m, WB, BC=_LANES, MP=MP)
                for i in range(k - 1, -1, -1):
                    _dot_kernel[(batch, nb)](
                        W, V[:, i], S, WB, VB, NP, BC=_LANES, MP=MP
                    )
                    _upd_kernel[(batch, nb)](
                        W, S, U[:, i], WB, NP, VB, BC=_LANES, MP=MP
                    )
        _out_kernel[(triton.cdiv(total, BT),)](OUT, W, total, m * n, n, WB, MP, BT=BT)

    return OUT[:total].view(shape)
