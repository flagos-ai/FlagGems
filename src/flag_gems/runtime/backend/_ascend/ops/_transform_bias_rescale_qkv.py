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

"""Ascend TLE implementation of ``_transform_bias_rescale_qkv``.

Reference semantics (FlagGems ``flag_gems.ops._transform_bias_rescale_qkv``)::

    qkv  : [B, S, 3 * H * D]     bias : [3 * H * D]
    q[b, h, s, d] = (qkv[b, s,           h*D + d] + bias[         h*D + d]) * scale
    k[b, h, s, d] =  qkv[b, s,     H*D + h*D + d] + bias[    H*D + h*D + d]
    v[b, h, s, d] =  qkv[b, s, 2 * H*D + h*D + d] + bias[2 * H*D + h*D + d]
    scale = 1 / sqrt(D)

Kernel shape: pure elementwise (vec-explicit).  One program owns one
``(segment, batch, head, seq-block)`` tile.  It stages a ``[BS, head_dim]``
source tile out of the strided ``[B, S, 3*H*D]`` input with a single bulk
``tle.dsa.copy`` GM->UB transfer, adds the per-column bias row and applies the
query scale in fp32, and writes the result tile back with a single bulk
``tle.dsa.copy`` UB->GM transfer.  Every input element is read once and every
output element is written once.

Exactly one UB tile buffer exists per program, so no two live tiles can ever
share one physical UB allocation.

Why the row block cap moves from 64 to 128
------------------------------------------
A copy descriptor here has to cover its tile with FULL extents: a partial leading
extent (``n_rows < BS``) is not correctly supported on this toolchain, so the row
block may only be a divisor of ``seq_len``.  That makes ``BS`` a correctness
invariant rather than a free knob -- a bounded ``@triton.autotune`` over ``BS``
was tried and it is unsafe, because it can select ``BS = 64/128`` for
``seq_len = 8/16`` and then emit exactly those partial-extent copies.
(Observed: 128/180 correctness failures, all on the small-``seq_len`` cases,
with 25-50% mismatched elements.)

Within that constraint one structural question is still open.  Round 1 caps the
block at 64, and every benchmark ``seq_len`` is a multiple of 64, so round 1
never varied the row block at all: it always ran 2+ programs per
``(segment, batch, head)`` for the small shapes.  Its own profile records the
smallest shape as scalar / fixed-overhead bound with the transfer pipes nearly
idle (``::0`` is ~1 us of transfer against a ~5 us latency).  With 48 vector
cores, ``::0`` at ``BS = 64`` launches ``3*2*8*2 = 96`` programs -- two waves --
while ``BS = 128`` launches 48, exactly one wave.  Amortising the per-program
prologue over twice the rows is the direct test of whether that residual cost is
program/wave-count dependent or a fixed per-launch cost no tiling can remove.
"""

import logging

import torch
import torch_npu  # noqa: F401  (must precede the triton import on Ascend)
import triton
import triton.experimental.tle as tle
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _tbqkv_tle_kernel(
    qkv_ptr,
    bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    scale,
    S,
    H,
    HD,
    DH,
    NBLK,
    BS: tl.constexpr,
    D: tl.constexpr,
):
    """``D`` is ``head_dim`` padded up to the next power of two; ``DH`` is the
    real ``head_dim`` (they coincide for every power-of-two head size)."""
    pid = tl.program_id(0)

    seg = pid % 3
    t1 = pid // 3
    sb = t1 % NBLK
    t2 = t1 // NBLK
    h = t2 % H
    b = t2 // H

    so = tl.arange(0, BS)
    co = tl.arange(0, D)
    row = sb * BS + so
    n_rows = tl.minimum(S - sb * BS, BS)
    cm = co < DH

    d_in = h * DH + co
    in_off = row[:, None] * (HD * 3) + d_in[None, :]

    dt = qkv_ptr.dtype.element_ty
    xt = tle.dsa.alloc([BS, D], dt, tle.dsa.ascend.UB)
    tle.dsa.copy(qkv_ptr + (b * (S * HD * 3) + seg * HD) + in_off, xt, [n_rows, DH])

    bias = tl.load(bias_ptr + seg * HD + d_in, mask=cm, other=0.0).to(tl.float32)
    z = tle.dsa.to_tensor(xt).to(tl.float32) + bias[None, :]
    if seg == 0:
        y = (z * scale).to(dt)
    else:
        y = z.to(dt)
    tle.dsa.to_buffer(y, tle.dsa.ascend.UB, xt)

    out_off = row[:, None] * DH + co[None, :]
    out_base = b * (H * S * DH) + h * (S * DH) + sb * BS * DH
    if seg == 0:
        tle.dsa.copy(xt, q_ptr + out_base + out_off, [n_rows, DH])
    elif seg == 1:
        tle.dsa.copy(xt, k_ptr + out_base + out_off, [n_rows, DH])
    else:
        tle.dsa.copy(xt, v_ptr + out_base + out_off, [n_rows, DH])


def _block_s(seq_len):
    """Largest power-of-two row block <= 128 that divides ``seq_len``.

    It must divide ``seq_len``: a copy descriptor has to cover its tile with full
    extents, and a partial leading extent is not supported here, so every block
    is whole.  The largest divisor that still keeps one UB tile small is chosen,
    which for the benchmark shapes (128/256/512/1024) is 128 rows and for the
    small correctness shapes (8/16/32) is the whole ``seq_len``.
    """
    bs = 128
    while bs > 1 and seq_len % bs != 0:
        bs //= 2
    return bs


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _transform_bias_rescale_qkv(qkv, qkv_bias, num_heads):
    """``_transform_bias_rescale_qkv`` backed by the TLE kernel."""
    logger.debug("GEMS _TRANSFORM_BIAS_RESCALE_QKV")
    if qkv.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops._transform_bias_rescale_qkv import (
            _transform_bias_rescale_qkv as _generic,
        )

        return _generic(qkv, qkv_bias, num_heads)

    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    bias = qkv_bias.contiguous()

    batch, seq_len, total_dim = qkv.shape
    num_heads = int(num_heads)
    hidden_dim = total_dim // 3
    head_dim = hidden_dim // num_heads

    q = torch.empty(
        (batch, num_heads, seq_len, head_dim), dtype=qkv.dtype, device=qkv.device
    )
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    if q.numel() == 0:
        return q, k, v

    bs = _block_s(seq_len)
    nblk = triton.cdiv(seq_len, bs)
    grid = (3 * batch * num_heads * nblk,)

    _tbqkv_tle_kernel[grid](
        qkv,
        bias,
        q,
        k,
        v,
        1.0 / (head_dim**0.5),
        seq_len,
        num_heads,
        hidden_dim,
        head_dim,
        nblk,
        BS=bs,
        D=triton.next_power_of_2(head_dim),
    )
    return q, k, v
