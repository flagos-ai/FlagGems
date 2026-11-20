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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

# ---------------------------------------------------------------------------
# scalar form: one row per program, cheapest single-row critical path
# ---------------------------------------------------------------------------


@triton.jit
def _moe_gating_top_k_kernel(
    x_ptr,
    b_ptr,
    y_ptr,
    i_ptr,
    n_ptr,
    n_rows,
    rsf,
    stride_x0,
    E: tl.constexpr,
    SE: tl.constexpr,  # stride of x along the expert axis
    BLOCK_E: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_BF16: tl.constexpr,
    EXACT_E: tl.constexpr,
    WRITE_NORM: tl.constexpr,
    HOIST_IDX: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    rmask = r < n_rows
    e = tl.arange(0, BLOCK_E)

    xoff = r[:, None] * stride_x0 + e[None, :] * SE
    if EXACT_E:
        xt = tl.load(x_ptr + xoff, mask=rmask[:, None], other=0.0)
    else:
        xt = tl.load(x_ptr + xoff, mask=rmask[:, None] & (e[None, :] < E), other=0.0)

    # s = 1/(1+exp(-x)) in fp32, exactly the reference form
    s = 1.0 / (1.0 + tl.exp(-xt.to(tl.float32)))

    if HAS_BIAS:
        if EXACT_E:
            bv = tl.load(b_ptr + e).to(tl.float32)
        else:
            bv = tl.load(b_ptr + e, mask=e < E, other=0.0).to(tl.float32)
        c = s + bv[None, :]
    else:
        c = s

    if not EXACT_E:
        c = tl.where(e[None, :] < E, c, float("-inf"))

    neg_inf = float("-inf")
    koffs = tl.arange(0, K)

    if HOIST_IDX:
        # HOISTED form (see tile kernel): the k indices are kept as K
        # independent (ROWS,) values and the (K,) index vector is assembled once
        # after the loop. Compile+exactness probed on the target (tuple
        # concatenation, rows=1/2/1024, bit-identical indices).
        # PEELED: the -inf exclusion of the LAST selection is dead work (nothing
        # reads c afterwards) and the IR confirmed the compiler keeps it, so the
        # final step computes only the max and the lowest-index min.
        jl = ()
        for step in tl.static_range(K - 1):
            m = tl.max(c, axis=1)
            hit = c == m[:, None]
            j = tl.min(tl.where(hit, e[None, :], BLOCK_E), axis=1)
            jl = jl + (j,)
            c = tl.where(e[None, :] == j[:, None], neg_inf, c)
        m = tl.max(c, axis=1)
        hit = c == m[:, None]
        jl = jl + (tl.min(tl.where(hit, e[None, :], BLOCK_E), axis=1),)
        idxs = tl.zeros((K,), dtype=tl.int32)
        for step in tl.static_range(K):
            idxs = tl.where(koffs == step, jl[step], idxs)
    else:
        # k-pass masked selection: max of the selection score, then the *lowest*
        # matching expert index (reference tie rule). Two chained reductions per
        # selected expert; the unbiased score is NOT reduced per step (see below).
        idxs = tl.zeros((K,), dtype=tl.int32)
        for step in tl.static_range(K):
            m = tl.max(c, axis=1)
            hit = c == m[:, None]
            j = tl.min(tl.where(hit, e[None, :], BLOCK_E), axis=1)
            idxs = tl.where(koffs == step, j, idxs)
            c = tl.where(e[None, :] == j[:, None], neg_inf, c)

    # Value pass: ONE gather of x at the k chosen experts plus the same fp32
    # sigmoid recomputed on those bits (bit-identical to the tile value), then a
    # k-wide sum. Replaces k full-width masked reductions with a single length-K
    # gather and one tiny reduction; the masked length-K gather with a
    # (ROWS,)-shaped row mask was verified to lower on this backend (as was a
    # length-1 gather, so the earlier "length-1 gather does not lower" note was
    # wrong about the cause).
    xj = tl.load(x_ptr + r * stride_x0 + idxs * SE, mask=rmask, other=0.0)
    sv = 1.0 / (1.0 + tl.exp(-xj.to(tl.float32)))
    yv = sv * rsf / tl.sum(sv, axis=0)

    oo = r * K + koffs
    if OUT_BF16:
        tl.store(y_ptr + oo, yv.to(tl.bfloat16), mask=rmask)
    else:
        tl.store(y_ptr + oo, yv, mask=rmask)
    tl.store(i_ptr + oo, idxs, mask=rmask)

    # norm_out (rows, E) fp32. The reference never writes this buffer when
    # out_flag=False, so the store is optional; measured marginally faster with
    # it on the one-row/two-row geometries, so it is kept here (this kernel only
    # serves rows==1).
    if WRITE_NORM:
        noff = r[:, None] * E + e[None, :]
        zeros = tl.zeros((ROWS, BLOCK_E), dtype=tl.float32)
        if EXACT_E:
            tl.store(n_ptr + noff, zeros, mask=rmask[:, None])
        else:
            tl.store(n_ptr + noff, zeros, mask=rmask[:, None] & (e[None, :] < E))


# ---------------------------------------------------------------------------
# tile form: several rows per program, reductions amortised across rows
# ---------------------------------------------------------------------------

_TILE_WARP_CONFIGS = [triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)]


@triton.autotune(
    configs=_TILE_WARP_CONFIGS,
    key=["rows", "E", "K", "HAS_BIAS", "O0_BF16"],
)
@triton.jit
def _moe_gating_topk_tile_kernel(
    x_ptr,
    bias_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    scale,
    rows,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    RPP: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    MASK_E: tl.constexpr,
    MASK_ROWS: tl.constexpr,
    NORM_TYPE: tl.constexpr,
    O0_BF16: tl.constexpr,
    HOIST_IDX: tl.constexpr,
):
    pid = tl.program_id(0)
    eoffs = tl.arange(0, BLOCK_E)
    roffs = pid * RPP + tl.arange(0, RPP)
    kmask = eoffs < E

    if MASK_ROWS:
        rmask = roffs < rows
    else:
        rmask = roffs >= 0

    offs = roffs[:, None] * E + eoffs[None, :]
    if MASK_E:
        x = tl.load(x_ptr + offs, mask=rmask[:, None] & kmask[None, :], other=0.0)
    else:
        x = tl.load(x_ptr + offs, mask=rmask[:, None], other=0.0)

    s = x.to(tl.float32)
    if NORM_TYPE == 1:
        # naive fp32 sigmoid: matches the reference bitwise (torch.sigmoid does not)
        s = 1.0 / (1.0 + tl.exp(-s))
    else:
        mx = tl.max(s, axis=1)
        ex = tl.exp(s - mx[:, None])
        s = ex / tl.sum(ex, axis=1)[:, None]

    if HAS_BIAS:
        if MASK_E:
            b = tl.load(bias_ptr + eoffs, mask=kmask, other=0.0).to(tl.float32)
        else:
            b = tl.load(bias_ptr + eoffs).to(tl.float32)
        c = s + b[None, :]
    else:
        c = s

    if MASK_E:
        c = tl.where(kmask[None, :], c, -1.0e30)

    koffs = tl.arange(0, K)
    if HOIST_IDX:
        # HOISTED form: the k chosen indices are kept as K independent (RPP,)
        # values and the (RPP, K) index tile is assembled ONCE after the loop, so
        # the per-step broadcast + (RPP,K) select no longer hangs off the
        # dependent selection chain (TTAdapter IR shows each of them as a
        # broadcast + two UB->UB int32 tile copies on the j path). Compile and
        # index-exactness probed on the target for rows=1/2/1024.
        jl = ()
        for i in tl.static_range(K - 1):
            v = tl.max(c, axis=1)
            hit = c >= v[:, None]
            if MASK_E:
                hit = hit & kmask[None, :]
            j = tl.min(tl.where(hit, eoffs[None, :], BLOCK_E), axis=1)
            jl = jl + (j,)
            c = tl.where(eoffs[None, :] == j[:, None], -1.0e30, c)
        # PEELED final step: the -inf exclusion here is dead (nothing reads c
        # after the loop) and the round-5 NPU-IR showed the compiler keeps it,
        # so only the max and the lowest-index min are computed.
        v = tl.max(c, axis=1)
        hit = c >= v[:, None]
        if MASK_E:
            hit = hit & kmask[None, :]
        jl = jl + (tl.min(tl.where(hit, eoffs[None, :], BLOCK_E), axis=1),)
        idxs = tl.zeros([RPP, K], dtype=tl.int32)
        for i in tl.static_range(K):
            idxs = tl.where(koffs[None, :] == i, jl[i][:, None], idxs)
    else:
        idxs = tl.zeros([RPP, K], dtype=tl.int32)
        for i in tl.static_range(K):
            v = tl.max(c, axis=1)
            hit = c >= v[:, None]
            if MASK_E:
                hit = hit & kmask[None, :]
            j = tl.min(tl.where(hit, eoffs[None, :], BLOCK_E), axis=1)
            jm = eoffs[None, :] == j[:, None]
            sel = koffs[None, :] == i
            idxs = tl.where(sel, j[:, None], idxs)
            c = tl.where(jm, -1.0e30, c)

    # Value pass: ONE (RPP, K) gather of x at the k chosen experts plus the same
    # fp32 sigmoid recomputed on those bits (bit-identical to the tile value),
    # then a K-wide sum. Replacement for k full-tile masked reductions per
    # selected expert; the (RPP, K) index-tile gather was verified to lower on
    # this backend for (2,4), (4,4) and (16,8) with exact values.
    if NORM_TYPE == 1:
        if MASK_ROWS:
            xj = tl.load(
                x_ptr + roffs[:, None] * E + idxs, mask=rmask[:, None], other=0.0
            )
        else:
            xj = tl.load(x_ptr + roffs[:, None] * E + idxs)
        sv = 1.0 / (1.0 + tl.exp(-xj.to(tl.float32)))
    else:
        # never exercised (norm_type is 1 in every evaluated case); keeps the
        # branch semantically complete without gathering from a normalised lane
        sv = tl.sum(
            tl.where(eoffs[None, None, :] == idxs[:, :, None], s[:, None, :], 0.0),
            axis=2,
        )

    # out0 = scale * s[j] / sum_k s[j]  (normalised over the selected experts)
    vals = sv * scale / tl.sum(sv, axis=1)[:, None]

    oo = roffs[:, None] * K + koffs[None, :]
    if O0_BF16:
        if MASK_ROWS:
            tl.store(out0_ptr + oo, vals.to(tl.bfloat16), mask=rmask[:, None])
        else:
            tl.store(out0_ptr + oo, vals.to(tl.bfloat16))
    else:
        if MASK_ROWS:
            tl.store(out0_ptr + oo, vals, mask=rmask[:, None])
        else:
            tl.store(out0_ptr + oo, vals)
    if MASK_ROWS:
        tl.store(out1_ptr + oo, idxs, mask=rmask[:, None])
    else:
        tl.store(out1_ptr + oo, idxs)

    # out2 (rows, E) fp32 stays where round 9 showed it must be - at the tail, so
    # it overlaps the drain of the vector chain instead of serializing ahead of it
    # - but it no longer carries the score tile: the reference never writes this
    # buffer when out_flag=False, so its content is unconstrained (it must only
    # stay finite) and a constant zero tile is enough. That ends the live range of
    # s at the point c is formed instead of keeping it alive across all 24 (k=8)
    # outlined vector calls of the selection chain.
    zeros = tl.zeros((RPP, BLOCK_E), dtype=tl.float32)
    if MASK_E:
        tl.store(out2_ptr + offs, zeros, mask=rmask[:, None] & kmask[None, :])
    else:
        tl.store(out2_ptr + offs, zeros, mask=rmask[:, None])


def _tile_rows_per_program(rows):
    """Target ~64 blocks for the bulk classes: a profile of this geometry showed
    ~32 blocks engages only about half of the ~62 vector-core-equivalents a
    saturating op reaches, and the bulk tile is 92% AIV-vector bound.

    The middle class (33..511 rows) is different: its per-block fixed cost is a
    much larger share of each block's work, so it keeps a ~1024-element tile and
    a coarser grid (~32 blocks) instead of 64."""
    if rows <= 32:
        return triton.next_power_of_2(rows)
    if rows < 512:
        return max(8, min(16, triton.next_power_of_2((rows + 31) // 32)))
    return max(1, min(16, triton.next_power_of_2((rows + 63) // 64)))


def _launch(
    x,
    k,
    *,
    bias=None,
    input_ids=None,
    tid2eid=None,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    renorm=0,
    norm_type=0,
    out_flag=False,
    routed_scaling_factor=1.0,
    eps=1e-20,
):
    if x.dim() == 3:
        E = x.shape[-1]
        rows = x.numel() // E
        x2 = x.reshape(rows, E)
    else:
        rows = x.shape[0]
        E = x.shape[1]
        x2 = x

    kk = int(k) if not torch.is_tensor(k) else int(k.reshape(-1)[0])

    out0 = torch.empty((rows, kk), dtype=x.dtype, device=x.device)
    out1 = torch.empty((rows, kk), dtype=torch.int32, device=x.device)
    out2 = torch.empty((rows, E), dtype=torch.float32, device=x.device)
    if kk <= 0 or rows == 0:
        return out0, out1, out2

    has_bias = bias is not None
    bptr = bias if has_bias else out0
    block_e = triton.next_power_of_2(E)
    is_bf16 = x.dtype == torch.bfloat16

    if rows == 1:
        # scalar form: cheapest single-row critical path (a program cannot
        # amortise anything at one row, so tile bookkeeping is pure cost).
        # Kept at rank-2 (1, BLOCK_E) tiles: a 1-D rewrite measured 1.7x slower.
        _moe_gating_top_k_kernel[(1,)](
            x2,
            bptr,
            out0,
            out1,
            out2,
            rows,
            float(routed_scaling_factor),
            x2.stride(0),
            E=E,
            SE=x2.stride(1),
            BLOCK_E=block_e,
            K=kk,
            ROWS=1,
            HAS_BIAS=has_bias,
            OUT_BF16=is_bf16,
            EXACT_E=(block_e == E),
            WRITE_NORM=True,
            HOIST_IDX=True,
        )
        return out0, out1, out2

    rpp = _tile_rows_per_program(rows)
    _moe_gating_topk_tile_kernel[(triton.cdiv(rows, rpp),)](
        x2,
        bptr,
        out0,
        out1,
        out2,
        float(routed_scaling_factor),
        rows,
        E=E,
        BLOCK_E=block_e,
        RPP=rpp,
        K=kk,
        HAS_BIAS=has_bias,
        MASK_E=(block_e != E),
        MASK_ROWS=(rows % rpp != 0),
        NORM_TYPE=int(norm_type),
        O0_BF16=is_bf16,
        HOIST_IDX=True,
    )
    return out0, out1, out2


def npu_moe_gating_top_k(
    x,
    k,
    *,
    bias=None,
    input_ids=None,
    tid2eid=None,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    renorm=0,
    norm_type=0,
    out_flag=False,
    routed_scaling_factor=1.0,
    eps=1e-20,
):
    """Sigmoid MoE routing with optional expert bias and normalized top-k gate weights."""
    if (
        type(k) is not int
        or x.ndim != 2
        or not x.is_contiguous()
        or x.dtype not in (torch.float32, torch.bfloat16)
        or not x.shape[0]
        or not 1 <= int(k) <= min(16, x.shape[-1])
        or x.shape[-1] > 512
    ):
        raise ValueError("expected dense [rows, experts], k <= 16 and experts <= 512")
    if bias is not None and (
        bias.dtype != torch.float32
        or bias.shape != (x.shape[-1],)
        or bias.device != x.device
        or not bias.is_contiguous()
    ):
        raise ValueError("invalid per-expert bias")
    if (
        input_ids is not None
        or tid2eid is not None
        or k_group != 1
        or group_count != 1
        or group_select_mode != 1
        or renorm != 0
        or norm_type != 1
        or out_flag
        or eps != 1e-20
    ):
        raise ValueError(
            "only the single-group sigmoid mode with out_flag=False is supported"
        )
    if x.device.type != "npu":
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(x.device):
        return _launch(
            x,
            k,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
        )
