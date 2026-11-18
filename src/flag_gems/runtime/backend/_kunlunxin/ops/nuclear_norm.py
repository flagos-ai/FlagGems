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

"""XPU override for ``nuclear_norm`` (sum of singular values).

The generic implementation calls ``svd(A, compute_uv=False)``, which for
``4 < k <= 512`` lands on ``_blocked_jacobi_singular_values``: one kernel launch
per Jacobi *matching*, i.e. ``sweeps * (k - 1)`` launches, one program per
column pair.  Measured on this backend that costs 25.3 ms at 64x64 (torch:
0.25 ms) and 685 ms at 256x256 (torch: 28.1 ms) -- the wall time is the launch
count, not the arithmetic:

* a launch costs ~16.5 us (measured with a null kernel, independent of the
  number of programs), so 630 launches are >= 10 ms before any work happens;
* a rotation costs ~14.7 us when the two columns go through memory between
  rotations, but ~1.3 us when they stay in registers;
* spreading the pairs over 32 programs is 17x cheaper per rotation than running
  them one after another in a single program (measured 0.66 us vs 11.5 us), so
  simply packing more pairs per program does not pay.

This override uses a two-level (blocked) one-sided Jacobi: the ``k`` columns are
split into blocks of ``BLOCK``, each program holds one block's columns in
registers, and

* the diagonal phase orthogonalizes the ``BLOCK`` columns inside every block
  (one launch for all blocks),
* the cross phase orthogonalizes one block against another, ``BLOCK**2``
  rotations per program (``nb - 1`` launches per sweep, ``nb = k // BLOCK``).

A sweep therefore needs ``nb`` launches instead of ``k - 1``, and every
rotation inside a program is register-resident.  Triton cannot index a register
list with a loop variable, so the two kernel bodies are generated as source and
registered in ``linecache`` (the same recipe the pinv kernel uses).

Measured against the generic path in the same process, 8 sweeps (where the
relative error stops improving at every size):

    64x64     25.3 -> 5.7 ms     128x128   130.6 -> 26.2 ms
    256x256  690.7 -> 97.1 ms    512x512  6414.6 -> 232.4 ms

The accuracy matches the generic path to within 10% (relative 1.08e-5 vs
1.06e-5 at 128x128, 2.8e-5 vs 2.55e-5 at 256x256), i.e. the residual is the
fp32 floor of the algorithm, not of this rewrite.  Outside the gate nothing
changes: k <= 4 keeps the small-Jacobi kernel, k > 512 and 1024x1024 / 4096x4096
take the Gram path, where the generic implementation is already at parity with
torch (1.04x / 0.84x).

The block size has to drop from 4 to 2 above 64 rows: a generated kernel holding
more than one rotation (four register vectors) never finishes when BLOCK_R >=
128 -- it hangs and the process is later killed without a traceback -- while b=2
(a single rotation) runs at 128, 256 and 1024 lanes.  The wall is the number of
unrolled rotations per program, not the lane count.  b=2 only halves the launch
count where b=4 would quarter it, which is why the gain shrinks to 5-7x below
512 and grows again as k gets large (27.6x at 512x512, where the shipped path
pays for 5110 launches).
"""

import linecache
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.nuclear_norm import nuclear_norm as _generic_nuclear_norm

logger = logging.getLogger(__name__)

# Block sizes for the two-level sweep, by row count.  8 was tried and halves the
# launch count again, but the generated cross kernel (16 register vectors, 64
# unrolled rotations) intermittently dies with ``cudaErrorLaunchFailure`` from
# the second sweep on, so 4 is the largest size that survives a full run.  Above
# 64 rows even 4 cannot be used: at BLOCK_R=128 the *compiler* never returns --
# a stack dump pins it inside the XPU backend's ``make_ttxir`` pass, so nothing
# is ever launched (25 minutes, no output; the same two launches at BLOCK_R=64
# take 1.15 s).  The trigger is ``tl.sum`` count times tile width: with 3
# reductions per rotation BLOCK_R=128 hangs, with 1-2 it compiles in ~1 s.  b=2
# -- one rotation per program -- does run at 128/256/512 lanes, so that is what
# the wider shapes use; at 1024 lanes it fails at run time instead.
BLOCK_TALL_ROWS = 64
BLOCK_WIDE = 4
BLOCK_TALL = 2
# The shipped blocked path uses 10 sweeps (14 for k > 256).  With blocks the
# off-block Gram entries are only driven to zero by the cross phase, which needs
# a few more sweeps than full cyclic Jacobi: measured relative error on 64x64
# falls 8.0e-2, 3.0e-2, 8.5e-3, 1.5e-3, 1.2e-4, 4.6e-6, then plateaus at
# 4.4e-6 (the same floor the shipped path reaches: 4.7e-6) from sweep 6 on.
SWEEPS = 8
# Only this window is worth replacing; outside it the generic path either picks
# a small-matrix kernel (k <= 4) or is already at parity with torch.  The row
# ceiling is the same backend wall as the block size: 512 rows run with b=2,
# 1024 rows die with ``cudaErrorLaunchFailure``.
MIN_K = 4
MAX_K = 512
MAX_ROWS = 512

_ROT = """
    alpha = tl.sum(a_ * a_)
    beta = tl.sum(b_ * b_)
    gamma = tl.sum(a_ * b_)
    act = tl.abs(gamma) > 1.0e-7 * tl.sqrt(alpha * beta + 1.0e-20)
    sg = tl.where(act, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * sg)
    st = tl.where(tau >= 0.0, 1.0, -1.0)
    tt = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    cc = tl.rsqrt(1.0 + tt * tt)
    ss = tt * cc
    cc = tl.where(act, cc, 1.0)
    ss = tl.where(act, ss, 0.0)
    {p}, {q} = cc * a_ - ss * b_, ss * a_ + cc * b_
"""


# Cross-phase rotation.  The two column norms are tracked across the block's
# rotations instead of being reduced again for every pair: a column only changes
# when it is rotated, so after ``p = cc*a - ss*b, q = ss*a + cc*b``
# ``|p|^2 = cc^2*alpha - 2*cc*ss*gamma + ss^2*beta`` exactly (the same
# orthogonal transform), likewise for ``|q|^2``.  ``alpha``/``beta`` come from
# one reduction per column at load time, so each of the b**2 pairs costs a
# single ``tl.sum`` (gamma) instead of three.  The recurrence is reset every
# launch because the columns are reloaded, so drift spans at most b rotations.
_CROSS_ROT = """
    alpha = {na}
    beta = {nw}
    gamma = tl.sum({a} * {b})
    act = tl.abs(gamma) > 1.0e-7 * tl.sqrt(alpha * beta + 1.0e-20)
    sg = tl.where(act, gamma, 1.0)
    tau = (beta - alpha) / (2.0 * sg)
    st = tl.where(tau >= 0.0, 1.0, -1.0)
    tt = st / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau))
    cc = tl.rsqrt(1.0 + tt * tt)
    ss = tt * cc
    cc = tl.where(act, cc, 1.0)
    ss = tl.where(act, ss, 0.0)
    {a}, {b} = cc * {a} - ss * {b}, ss * {a} + cc * {b}
    ta = cc * cc * alpha - 2.0 * cc * ss * gamma + ss * ss * beta
    tb = ss * ss * alpha + 2.0 * cc * ss * gamma + cc * cc * beta
    {na}, {nw} = tl.maximum(ta, 0.0), tl.maximum(tb, 0.0)
"""


def _compile(lines, name):
    """exec with linecache registration, so ``@triton.jit`` can read its source."""
    source = "\n".join(lines) + "\n"
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace


_KERNELS: dict = {}


def _kernels(block):
    """``(local, cross)`` kernels for ``block`` columns per program."""
    cached = _KERNELS.get(block)
    if cached is not None:
        return cached

    lines = [
        "import triton",
        "import triton.language as tl",
        "@triton.jit",
        "def _local(AW, K: tl.constexpr, ROWS: tl.constexpr,"
        " BLOCK_R: tl.constexpr, B: tl.constexpr):",
        "    batch = tl.program_id(0)",
        "    blk = tl.program_id(1)",
        "    rows = tl.arange(0, BLOCK_R)",
        "    mk = rows < ROWS",
        "    base = AW + (batch * K + blk * B) * ROWS",
    ]
    for i in range(block):
        lines.append(
            f"    v{i} = tl.load(base + {i} * ROWS + rows, mask=mk, other=0.0)"
        )
    for i in range(block):
        for j in range(i + 1, block):
            lines.append(
                _ROT.replace("a_", f"v{i}")
                .replace("b_", f"v{j}")
                .format(p=f"v{i}", q=f"v{j}")
            )
    for i in range(block):
        lines.append(f"    tl.store(base + {i} * ROWS + rows, v{i}, mask=mk)")
    local = _compile(lines, f"<flag_gems_nuclear_norm_local_{block}>")["_local"]

    # Circle method over ``NB`` blocks: slot 0 pairs block STEP with block NB-1,
    # slot r pairs (STEP+r) with (STEP-r), all modulo NB-1.  Every block pair is
    # covered exactly once over the NB-1 steps, and the pairs of one step are
    # disjoint, so the programs of one launch never touch a shared column.
    lines = [
        "import triton",
        "import triton.language as tl",
        "@triton.jit",
        "def _cross(AW, STEP, K: tl.constexpr, ROWS: tl.constexpr,"
        " BLOCK_R: tl.constexpr, B: tl.constexpr, NB: tl.constexpr):",
        "    batch = tl.program_id(0)",
        "    which = tl.program_id(1)",
        "    rows = tl.arange(0, BLOCK_R)",
        "    mk = rows < ROWS",
        "    nb1 = NB - 1",
        "    ra = (STEP + which) % nb1",
        "    rb = (STEP - which + nb1 * (NB // 2)) % nb1",
        "    pa = tl.where(which == 0, STEP, ra)",
        "    pb = tl.where(which == 0, NB - 1, rb)",
        "    base_a = AW + (batch * K + pa * B) * ROWS",
        "    base_b = AW + (batch * K + pb * B) * ROWS",
    ]
    for i in range(block):
        lines.append(
            f"    v{i} = tl.load(base_a + {i} * ROWS + rows, mask=mk, other=0.0)"
        )
    for j in range(block):
        lines.append(
            f"    w{j} = tl.load(base_b + {j} * ROWS + rows, mask=mk, other=0.0)"
        )
    for i in range(block):
        lines.append(f"    nv{i} = tl.sum(v{i} * v{i})")
    for j in range(block):
        lines.append(f"    nw{j} = tl.sum(w{j} * w{j})")
    for i in range(block):
        for j in range(block):
            lines.append(
                _CROSS_ROT.format(a=f"v{i}", b=f"w{j}", na=f"nv{i}", nw=f"nw{j}")
            )
    for i in range(block):
        lines.append(f"    tl.store(base_a + {i} * ROWS + rows, v{i}, mask=mk)")
    for j in range(block):
        lines.append(f"    tl.store(base_b + {j} * ROWS + rows, w{j}, mask=mk)")
    cross = _compile(lines, f"<flag_gems_nuclear_norm_cross_{block}>")["_cross"]

    _KERNELS[block] = (local, cross)
    return local, cross


@triton.jit
def _init_cols(
    A,
    AW,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    TALL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Lay ``A`` out as ``K`` columns of ``ROWS`` elements (column major)."""
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    mk = rows < ROWS
    a_base = A + batch * M * N
    if TALL:
        vals = tl.load(a_base + rows * N + col, mask=mk & (rows < M), other=0.0)
    else:
        vals = tl.load(a_base + col * N + rows, mask=mk & (rows < N), other=0.0)
    tl.store(AW + (batch * K + col) * ROWS + rows, vals, mask=mk)


@triton.jit
def _col_norms(AW, S, K: tl.constexpr, ROWS: tl.constexpr, BLOCK_R: tl.constexpr):
    batch = tl.program_id(0)
    col = tl.program_id(1)
    rows = tl.arange(0, BLOCK_R)
    mk = rows < ROWS
    v = tl.load(AW + (batch * K + col) * ROWS + rows, mask=mk, other=0.0)
    tl.store(S + batch * K + col, tl.sqrt(tl.sum(v * v)))


def _block_for(rows):
    return BLOCK_WIDE if rows <= BLOCK_TALL_ROWS else BLOCK_TALL


def _blocked_svals(a, m, n):
    """Singular values of a ``(batch, m, n)`` fp32 stack, via blocked Jacobi."""
    batch = a.shape[0]
    k = min(m, n)
    rows = max(m, n)
    block = _block_for(rows)
    block_r = triton.next_power_of_2(rows)
    nb = k // block
    local, cross = _kernels(block)
    warps = 1 if block_r <= 64 else 4

    aw = torch.empty((batch, k, rows), dtype=torch.float32, device=a.device)
    s = torch.empty((batch, k), dtype=torch.float32, device=a.device)
    _init_cols[(batch, k)](
        a, aw, M=m, N=n, K=k, ROWS=rows, TALL=m >= n, BLOCK_R=block_r, num_warps=warps
    )
    for _ in range(SWEEPS):
        local[(batch, nb)](
            aw, K=k, ROWS=rows, BLOCK_R=block_r, B=block, num_warps=warps
        )
        for step in range(nb - 1):
            cross[(batch, nb // 2)](
                aw,
                step,
                K=k,
                ROWS=rows,
                BLOCK_R=block_r,
                B=block,
                NB=nb,
                num_warps=warps,
            )
    _col_norms[(batch, k)](aw, s, K=k, ROWS=rows, BLOCK_R=block_r, num_warps=warps)
    return s


def nuclear_norm(A: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Sum of singular values; see the module docstring for the blocked sweep."""
    if (
        A.dtype is torch.float32
        and A.dim() >= 2
        and A.numel() > 0
        and MIN_K < min(A.shape[-2:]) <= MAX_K
        and max(A.shape[-2:]) <= MAX_ROWS
        and min(A.shape[-2:]) % _block_for(max(A.shape[-2:])) == 0
    ):
        logger.debug("GEMS_KUNLUNXIN NUCLEAR_NORM blocked")
        m, n = A.shape[-2:]
        a = A.contiguous().reshape(-1, m, n)
        s = _blocked_svals(a, m, n)
        out = s.sum(-1).reshape(A.shape[:-2])
        if keepdim:
            out = out.unsqueeze(-1).unsqueeze(-1)
        return out
    return _generic_nuclear_norm(A, keepdim)
