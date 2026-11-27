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

"""Kunlunxin override of `aten::_scaled_dot_product_attention_math`.

Why the generic kernel cannot be used here
------------------------------------------
* It feeds masked 2-D loads straight into `tl.dot`, which fails to lower in
  `ConvertTritonSDNNToLLVM`.
* It uses a float select on the softmax denominator
  (`tl.where(l == 0.0, 1.0, l)`), which the sibling `_for_mps` generic kernel
  reports as `LLVM ERROR: Invalid selectType_: cannot invert non-select EW
  funct`.
* Even after removing those, it miscompiles a `tl.dot` whose operand is a
  *register-derived* tile: the PV product `dot(exp2(dot(q, k^T)), v)` came back
  with order-1 relative error for fp16/bf16/fp32 in a standalone probe, while
  the QK dot itself was exact.  Only load-rooted dot operands are reliable.
* A row reduction of a `tl.dot` result is miscompiled as well: `tl.max(qk, 1)`,
  `tl.sum(p, 1)`, and the same reductions over the transposed tile all come
  back 0 or in the wrong lane (measured), so the softmax denominator collapses
  and the row max is lost.  The identical reduction over a *load-rooted* tile
  is exact.  The scores are therefore staged through memory and read back
  before they are reduced.
* A single kernel that keeps both the softmax-statistics loop and the
  weight-materialisation loop was also miscompiled (measured), so the passes
  are split; each kernel below was validated in isolation.
* Fusing the normalisation into the exponential is miscompiled as well: in
  ``p = tl.exp(...) * inv_l`` the compiler folds the two into one expression and
  the masked lanes come back as ``exp(-16) * inv_l`` instead of ``0`` (measured
  for every clamp value from -30 to -1e4).  The same product written as two
  statements is still folded, so the scaling is a separate kernel.
* A 1-D load of a per-row value is miscompiled inside that separate kernel:
  ``tl.load(L + off_z * Q_CTX + offs_m)`` splats a single element of the row
  across the whole tile (measured), so the denominator is recomputed from the
  weights with a load-rooted reduction instead of being read back.
* A kernel that loads and stores the same tile returns the unnormalised values
  (measured), so every pass here is out of place.

Design (five kernels)
---------------------
1. ``_sdpa_scores_kernel``  — QK scores (masked) → the score matrix.
2. ``_sdpa_stats_kernel``   — score matrix → per-row softmax max (m).
3. ``_sdpa_weights_kernel`` — score matrix + stats → unnormalised weights.
4. ``_sdpa_scale_kernel``   — unnormalised weights → normalised weights, by the
   row denominator it recomputes from the weights themselves.
5. ``_sdpa_pv_kernel``      — ``out = weights @ V`` as a tiled GEMM whose
   operands are both loads.

Every `tl.dot` operand is an unmasked, affine, in-bounds load and every tile is
full: the host pads K/V (and the attention mask) up to a multiple of the key
tile, so no kernel needs a clamped block base.  That clamp — a scalar
`tl.minimum` on the key-loop induction variable — hangs the compiler on this
backend even with no other arithmetic in the loop, so it is avoided rather than
worked around.  Masks are applied with `tl.where` to the *dot result*, never to
an operand.  Both dots are 2-arg with an explicit accumulator.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Softmax constants, re-declared as locals inside each kernel (Triton only
# resolves module globals that are `tl.constexpr` instances).
#   NEG    -- sentinel for "no key column here"; finite so `m_i - m_new` stays
#             finite for rows that never see a real column
#   L_MIN  -- seed for the online denominator, so it is never 0


@libentry()
@triton.jit
def _sdpa_scores_kernel(
    Q,
    K,
    ATTN_MASK,
    Scores,
    sm_scale,
    CAUSAL_DIAG,
    stride_qz: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_mz: tl.constexpr,
    stride_mm: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_sz: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    K_GROUP,
    M_OFFSET,
):
    """QK^T * sm_scale, masked, written out as the score matrix.

    Nothing here is reduced: the scores are read back by `_sdpa_stats_kernel`
    because a row reduction of a dot result is miscompiled on this backend.
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    # GQA/broadcast: fold the query head onto its K group (the index is
    # precomputed on the host; flagtree's inliner cannot lower arith.divsi).
    k_base = K + tl.load(K_GROUP + off_z) * stride_kz

    # The only scalar clamp left, and it is on the query axis, outside the key
    # loop: the last query block is shifted back so it reads real rows
    # (overlapping the previous block) instead of out-of-range lanes.
    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    if BLOCK_D == HEAD_DIM:
        rd = tl.arange(0, BLOCK_D)
    else:
        # Non-power-of-two head dims keep the addresses in-bounds and zero the
        # padding lanes by arithmetic (a per-lane select on a dot operand is
        # miscompiled on this backend).
        rd = tl.minimum(tl.arange(0, BLOCK_D), HEAD_DIM - 1)

    query = tl.load(
        Q + off_z * stride_qz + offs_m[:, None] * stride_qm + rd[None, :] * stride_qk
    )
    if BLOCK_D != HEAD_DIM:
        dkeep = (tl.arange(0, BLOCK_D) < HEAD_DIM).to(query.dtype)
        query = query * dkeep[None, :]

    NEG = -1.0e30
    diag = CAUSAL_DIAG
    s_base = Scores + off_z * stride_sz

    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        key = tl.load(k_base + offs_n[:, None] * stride_kn + rd[None, :] * stride_kk)
        if BLOCK_D != HEAD_DIM:
            key = key * dkeep[None, :]

        qk = tl.dot(query, tl.trans(key), out_dtype=tl.float32, allow_tf32=False)
        qk = qk * sm_scale

        if HAS_ATTN_MASK:
            m_ptrs = (
                ATTN_MASK
                + tl.load(M_OFFSET + off_z)
                + offs_m[:, None] * stride_mm
                + offs_n[None, :] * stride_mn
            )
            qk = qk + tl.load(m_ptrs).to(tl.float32)

        # The host pads the key axis with -inf columns, so every tile is full
        # and padded lanes already carry qk = -inf: exp gives them p = 0 with no
        # mask at all.  A `tl.where` on the dot result does not lower on this
        # backend, so the only select left is the causal one.
        if IS_CAUSAL:
            qk = tl.where(offs_n[None, :] <= offs_m[:, None] + diag, qk, NEG)

        tl.store(
            s_base + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
            qk,
        )


@libentry()
@triton.jit
def _sdpa_stats_kernel(
    Scores,
    M,
    stride_sz: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
):
    """Online softmax row max `m`.

    One pass over the (cached) score matrix, because a *1-D* loop-carried
    tensor is rejected by this backend -- `tt.addptr`/`arith.maxnumf` come back
    with "requires the same encoding for all operands" as soon as a reduce
    result is carried into the next iteration.  A 2-D accumulator carries fine,
    so the running max is kept as a ``[BLOCK_M, BLOCK_N]`` tile and reduced
    once, after the loop.

    Only the row max is produced here.  The denominator is recomputed by
    `_sdpa_scale_kernel` from the weights themselves: a 1-D load of a per-row
    value (`tl.load(L + off_z * Q_CTX + offs_m)`) is miscompiled in that kernel
    and splats a single element across the tile, so the value is never read
    back from memory.
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    NEG = -1.0e30
    s_base = Scores + off_z * stride_sz

    # Elementwise running max.  Seeded at NEG, so a row whose every lane is
    # padding (-inf) keeps the finite seed and p = exp(-inf - NEG) = 0 below,
    # which is what PyTorch returns for a fully masked row.
    m_tile = tl.full([BLOCK_M, BLOCK_N], NEG, dtype=tl.float32)
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        s = tl.load(
            s_base + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        ).to(tl.float32)
        m_tile = tl.maximum(m_tile, s)
    m_i = tl.max(m_tile, 1)

    tl.store(M + off_z * Q_CTX + offs_m, m_i)


@libentry()
@triton.jit
def _sdpa_weights_kernel(
    Scores,
    Weights,
    M,
    stride_sz: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_wz: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
):
    """Materialise the unnormalised weights ``exp(s - m)``.

    The result goes to its own buffer: writing a tile back over the one it was
    just loaded from miscompiles on this backend (measured: the pass returns the
    unnormalised values), and keeping the normalised weights in a separate
    buffer is also what leaves a load-rooted tile for `_sdpa_pv_kernel`, the
    only form of `tl.dot` operand this backend lowers correctly.

    The division by the row denominator is *not* done here: the compiler folds
    ``tl.exp(...) * inv_l`` into a single expression whose masked lanes come back
    as ``exp(-16) * inv_l`` rather than 0, so `_sdpa_scale_kernel` does it.
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    m_i = tl.load(M + off_z * Q_CTX + offs_m)

    s_base = Scores + off_z * stride_sz
    w_base = Weights + off_z * stride_wz
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        s = tl.load(
            s_base + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        ).to(tl.float32)
        # The clamp is on the exponent, not on the score: a masked lane carries
        # -inf, and `tl.exp` of -inf comes back NaN on this backend.  Clamping
        # the difference at -1e4 makes exp underflow instead, which is the same
        # value, and leaves every real lane untouched.
        p = tl.exp(tl.maximum(s - m_i[:, None], -1.0e4))
        # `tl.exp` does not underflow to 0 on this backend: every very negative
        # argument comes back as 2**-23 (measured for every clamp from -30 to
        # -1e4), which the 1/l in the scale pass then amplifies to inf for a row
        # whose every column is masked.  Zero those lanes explicitly.
        p = tl.where(s < -1.0e30, 0.0, p)
        tl.store(
            w_base + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
            p.to(Weights.dtype.element_ty),
        )


@libentry()
@triton.jit
def _sdpa_scale_kernel(
    Unnorm,
    Weights,
    stride_uz: tl.constexpr,
    stride_um: tl.constexpr,
    stride_un: tl.constexpr,
    stride_wz: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
):
    """Normalise the weight tile by its row denominator.

    A separate kernel because the compiler folds ``tl.exp(...) * inv_l`` into
    one expression and miscompiles the masked lanes; with the exponential in a
    different kernel the plain divide is exact.  Out of place for the same
    reason: a kernel that loads and stores the same tile returns the
    unnormalised values.

    The denominator is recomputed from the weights rather than read back from
    the stats kernel: a 1-D load of a per-row value is miscompiled here (it
    splats one element across the whole tile), while a reduction over the
    load-rooted weight tile is exact.
    """
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    u_base = Unnorm + off_z * stride_uz
    w_base = Weights + off_z * stride_wz

    l_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        w = tl.load(
            u_base + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
        )
        l_tile = l_tile + w
    # Seeded above zero so the divide below stays finite; the weights of a fully
    # masked row are already 0, so the seed does not change the result.
    l_i = tl.sum(l_tile, 1) + 1.0e-6

    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        w = tl.load(
            u_base + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
        )
        tl.store(
            w_base + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
            (w / l_i[:, None]).to(Weights.dtype.element_ty),
        )


@libentry()
@triton.jit
def _sdpa_pv_kernel(
    Weights,
    V,
    Out,
    stride_wz: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT_M: tl.constexpr,
    V_GROUP,
):
    """out[m, :] = sum_n weights[m, n] * v[n, :] — both dot operands are loads."""
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    v_base = V + tl.load(V_GROUP + off_z) * stride_vz

    if EXACT_M:
        m_base = start_m * BLOCK_M
    else:
        m_base = tl.minimum(start_m * BLOCK_M, Q_CTX - BLOCK_M)
    offs_m = m_base + tl.arange(0, BLOCK_M)

    if BLOCK_DV == VALUE_DIM:
        rdv = tl.arange(0, BLOCK_DV)
    else:
        rdv = tl.minimum(tl.arange(0, BLOCK_DV), VALUE_DIM - 1)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    # BLOCK_N divides the padded KV_CTX (chosen on the host), so every tile here
    # is a full, in-bounds affine load -- no clamp and no mask on the operands.
    for start_n in range(0, KV_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        w_tile = tl.load(
            Weights
            + off_z * stride_wz
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn
        )
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + rdv[None, :] * stride_vk
        )
        acc += tl.dot(w_tile, v_tile, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        Out
        + off_z * stride_oz
        + offs_m[:, None] * stride_om
        + rdv[None, :] * stride_ok,
        acc.to(Out.dtype.element_ty),
    )


def _pick_block(n, cap):
    """Largest power of two <= n, capped at `cap`."""
    block = 1
    while block * 2 <= n:
        block *= 2
    return min(cap, block)


def _pad_kv(tensor, padded_len):
    """Append zero rows along the sequence axis (dim 1) up to `padded_len`.

    `tensor` is the folded 3-D view ``(batch * heads, seq, dim)``.  Only the
    values matter for the padded lanes: the mask supplies -inf there, so the
    kernel never uses them.
    """
    if tensor.shape[1] == padded_len:
        return tensor
    pad = torch.zeros(
        (tensor.shape[0], padded_len - tensor.shape[1], tensor.shape[2]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    # Host-side K/V padding: a pure data move that feeds the device kernels,
    # which then need no clamped block base.
    return torch.cat((tensor, pad), dim=1)  # allow-torch-compute: padding


def _sdpa_math_impl(
    query,
    key,
    value,
    attn_mask=None,
    is_causal=False,
    scale=None,
    causal_diag=None,
):
    """Shared host path for the math and math_for_mps overrides."""
    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, heads, q_seq_len, head_dim = query.shape
    kv_seq_len = key.shape[2]
    value_dim = value.shape[3]
    key_heads = key.shape[1]
    value_heads = value.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    else:
        scale = float(scale)

    if causal_diag is None:
        causal_diag = 0

    # K/V may have fewer heads than Q (GQA) or a single broadcast head; fold
    # each query head onto the corresponding K/V group by dividing the folded
    # (batch, head) index by the group size, exactly as the generic kernel does.
    key_group_size = heads // key_heads if heads else 1
    value_group_size = heads // value_heads if heads else 1

    bh = batch * heads
    q2 = query.view(bh, q_seq_len, head_dim)

    BLOCK_M = _pick_block(q_seq_len, 64)
    BLOCK_N = _pick_block(kv_seq_len, 64)
    # Pad the key axis up to a whole number of tiles with -inf columns.  The
    # kernels then need no clamped block base (a scalar `tl.minimum` on the key
    # loop hangs this compiler) and no column mask (exp2(-inf) = 0 gives the
    # padded lanes p = 0 for free); the extra columns are sliced off the
    # returned weights.
    padded_kv_len = triton.cdiv(kv_seq_len, BLOCK_N) * BLOCK_N

    k2 = _pad_kv(key.view(batch * key_heads, kv_seq_len, head_dim), padded_kv_len)
    v2 = _pad_kv(
        value.view(batch * value_heads, kv_seq_len, value_dim), padded_kv_len
    )

    # Precompute the folded K/V group index for each (batch, head) slot on the
    # host.  flagtree's inliner cannot lower arith.divsi inside these kernels,
    # so the per-head division is done here and the result is loaded instead.
    k_group = (torch.arange(bh, dtype=torch.int32) // key_group_size).to(query.device)
    v_group = (torch.arange(bh, dtype=torch.int32) // value_group_size).to(query.device)

    out = torch.empty_like(query)
    o2 = out.view(bh, q_seq_len, value_dim)
    # `scores` stages the QK matrix between the kernel that produces it and the
    # one that reduces it; `weights` is the normalised result and the returned
    # attention weights.  They are separate buffers because the normalisation
    # pass miscompiles when it writes back over the fp16 scores in place.
    scores = torch.empty(
        (bh, q_seq_len, padded_kv_len), dtype=torch.float32, device=query.device
    )
    unnorm = torch.empty(
        (bh, q_seq_len, padded_kv_len), dtype=query.dtype, device=query.device
    )
    weights = torch.empty(
        (bh, q_seq_len, padded_kv_len), dtype=query.dtype, device=query.device
    )
    stats_m = torch.empty((bh, q_seq_len), dtype=torch.float32, device=query.device)

    # A mask is needed even when the caller passed none, because the padded key
    # columns have to read as -inf rather than as real keys.
    has_attn_mask = attn_mask is not None or padded_kv_len != kv_seq_len
    if attn_mask is not None:
        assert attn_mask.dtype != torch.bool, "boolean attn_mask is not supported"
        am = attn_mask.expand(batch, heads, q_seq_len, kv_seq_len)
    else:
        am = None
    if padded_kv_len != kv_seq_len:
        if am is None:
            am = torch.zeros(
                (batch, heads, q_seq_len, kv_seq_len),
                dtype=query.dtype,
                device=query.device,
            )
        # -inf on the padding columns: exp2 turns them into p = 0.  Host-side
        # mask padding, a pure data move.
        am = torch.nn.functional.pad(  # allow-torch-compute: padding
            am, (0, padded_kv_len - kv_seq_len), value=-float("inf")
        )
    if has_attn_mask:
        # The kernel folds (batch, head) into one grid axis, so the mask offset
        # for a folded slot is `b * stride(0) + h * stride(1)` -- not
        # `slot * stride(1)`, which only holds for a single batch.  The division
        # and modulo are done here because flagtree cannot lower arith.divsi.
        slots = torch.arange(bh, dtype=torch.int32, device=query.device)
        m_offset = (slots // heads) * am.stride(0) + (slots % heads) * am.stride(1)
        m_strides = (0, am.stride(2), am.stride(3))
        mask_buf = am
    else:
        mask_buf = query  # unused; HAS_ATTN_MASK guards every access
        m_offset = query  # unused
        m_strides = (0, 0, 0)

    # The scale pass holds an fp32 tile, its running sum and a cast in one
    # kernel; the full key tile overflows uni_sram, so it walks the key axis in
    # narrower slices.
    scale_block_n = min(BLOCK_N, 16)

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(value_dim)

    grid = (triton.cdiv(q_seq_len, BLOCK_M), bh)

    # Every pass walks the whole padded key axis, so the padding columns of the
    # score matrix are written (as -inf) and normalised to 0 as well.
    kv_ctx = padded_kv_len
    exact_m = q_seq_len % BLOCK_M == 0

    with torch_device_fn.device(query.device):
        _sdpa_scores_kernel[grid](
            q2,
            k2,
            mask_buf,
            scores,
            scale,
            causal_diag,
            q2.stride(0),
            q2.stride(1),
            q2.stride(2),
            k2.stride(0),
            k2.stride(1),
            k2.stride(2),
            m_strides[0],
            m_strides[1],
            m_strides[2],
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            q_seq_len,
            kv_ctx,
            HEAD_DIM=head_dim,
            BLOCK_D=BLOCK_D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EXACT_M=exact_m,
            IS_CAUSAL=is_causal,
            HAS_ATTN_MASK=has_attn_mask,
            K_GROUP=k_group,
            M_OFFSET=m_offset,
        )
        _sdpa_stats_kernel[grid](
            scores,
            stats_m,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            q_seq_len,
            kv_ctx,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EXACT_M=exact_m,
        )
        _sdpa_weights_kernel[grid](
            scores,
            unnorm,
            stats_m,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            unnorm.stride(0),
            unnorm.stride(1),
            unnorm.stride(2),
            q_seq_len,
            kv_ctx,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EXACT_M=exact_m,
        )
        _sdpa_scale_kernel[grid](
            unnorm,
            weights,
            unnorm.stride(0),
            unnorm.stride(1),
            unnorm.stride(2),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            q_seq_len,
            kv_ctx,
            BLOCK_M=BLOCK_M,
            BLOCK_N=scale_block_n,
            EXACT_M=exact_m,
        )
        _sdpa_pv_kernel[grid](
            weights,
            v2,
            o2,
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            v2.stride(0),
            v2.stride(1),
            v2.stride(2),
            o2.stride(0),
            o2.stride(1),
            o2.stride(2),
            q_seq_len,
            kv_ctx,
            VALUE_DIM=value_dim,
            BLOCK_DV=BLOCK_DV,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            EXACT_M=exact_m,
            V_GROUP=v_group,
        )

    weights = weights.view(bh, q_seq_len, padded_kv_len)[:, :, :kv_seq_len]
    return out, weights.view(batch, heads, q_seq_len, kv_seq_len)


def _scaled_dot_product_attention_math(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    dropout_mask=None,
    *,
    scale=None,
    enable_gqa=False,
):
    """Kunlunxin override: math SDPA returning (output, attention weights)."""
    logger.debug("GEMS_KUNLUNXIN _SCALED_DOT_PRODUCT_ATTENTION_MATH")
    assert dropout_p == 0.0, "dropout_p != 0.0 is not supported"
    assert query.dim() == 4, "expected 4D (batch, heads, seq_len, head_dim) tensors"

    heads = query.shape[1]
    key_heads = key.shape[1]
    value_heads = value.shape[1]

    if enable_gqa:
        if (
            key_heads == 0
            or value_heads == 0
            or heads % key_heads != 0
            or heads % value_heads != 0
        ):
            raise RuntimeError("query heads must be divisible by key and value heads")
    elif key_heads not in (1, heads) or value_heads not in (1, heads):
        raise RuntimeError("key and value heads must match or broadcast to query heads")

    return _sdpa_math_impl(
        query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )