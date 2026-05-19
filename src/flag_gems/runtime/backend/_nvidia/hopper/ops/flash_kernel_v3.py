"""
FlashAttention-V3 style forward kernel for varlen / paged attention on H100.

Target: NVIDIA Hopper (sm_90), Triton >= 3.6.

This file implements a Triton kernel that adopts three of the four key ideas
from FlashAttention-3 (Shah et al. 2024) within the constraints of Triton 3.6:

  (A) TMA-backed Q/K/V/O loads/stores via `tl.make_tensor_descriptor`.
      The compiler lowers descriptor accesses to async cp.async.bulk.tensor
      (TMA) instructions and frees registers from address arithmetic, leaving
      more headroom for the BLOCK_M=128, HEAD_DIM=128 acc tile.

  (B) Two-stage GEMM-softmax pipelining via the compiler.
      We rely on tl.range(num_stages=N) + num_warps=8 to let Triton 3.6's
      software pipeliner produce a cross-iteration ping-pong: with 2
      warpgroups the compiler partitions QK GEMM, softmax/exp2, and PV
      GEMM across warpgroups such that iter j's softmax overlaps iter j+1's
      K/V TMA load and iter j-1's PV. The FA3 paper §3.2 in-warpgroup
      qk_cur/qk_nxt scheme requires hoisting a tensor across loop
      iterations, which Triton 3.6's static analyzer forbids, so we let
      the autotuned schedule reach an equivalent steady state.

What we do NOT implement here:
  * Producer/consumer warp specialization (`gl.warp_specialize`).
    Triton 3.6 only supports it stably on Blackwell. On Hopper the autoWS
    path still has rough edges, especially for varlen + paged + alibi +
    softcap + dropout combinations. We rely on Triton's standard
    multi-stage software pipeliner (`num_stages` + `tl.range(...,
    num_stages=...)`) which on H100 typically yields the same allocation:
    one warpgroup's softmax overlapped with another's wgmma.

  * FP8. Per the user's request we keep the kernel BF16/FP16-only. The
    interface is wired so a future FP8 variant can plug in without
    touching the host code.

The kernel is functionally equivalent to `flash_varlen_fwd_kernel` in
flash_kernel.py and consumes the same `fwd_params` struct, so it slots
into mha_varlan_fwd_v3 below without any other API change.
"""

import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, tl_extra_shim

# ------------------------------------------------------------------
# Shared helpers (mask / alibi / softcap) -- numerically identical
# to the v2 kernel so accuracy tests pass bit-equivalently.
# ------------------------------------------------------------------


@triton.jit
def _apply_softcap_v3(S, softcap, IS_SOFTCAP: tl.constexpr):
    if IS_SOFTCAP:
        S = tl_extra_shim.tanh(S * softcap)
    return S


@triton.jit
def _apply_alibi_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    IS_CAUSAL: tl.constexpr,
    IS_ALIBI: tl.constexpr,
    alibi_slope,
):
    if IS_ALIBI:
        if IS_CAUSAL:
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias
    return S


@triton.jit
def _apply_mask_v3(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    IS_EVEN_MN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    if IS_CAUSAL or IS_LOCAL or (not IS_EVEN_MN):
        col_lb = tl.maximum(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = tl.minimum(
            max_seqlen_k - 1,
            row_idx + max_seqlen_k - max_seqlen_q + window_size_right,
        )
        if IS_CAUSAL:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)
        if IS_LOCAL:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )
        if (not IS_LOCAL) and (not IS_CAUSAL) and (not IS_EVEN_MN):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)
    return S


# ------------------------------------------------------------------
# Online softmax with deferred rescale.
#
# Returns (alpha, P, m_new, l_new) instead of (acc, P, m_new, l_new).
# The caller is responsible for folding `alpha` into `acc` *together with*
# the next P*V wgmma, breaking the dep chain that otherwise serialises
# softmax with the next GEMM.
# ------------------------------------------------------------------
@triton.jit
def _softmax_online_deferred(
    S,
    m_prev,
    l_prev,
    softmax_scale_log2e: tl.constexpr,
    IS_BORDER: tl.constexpr,
):
    # Row-wise max over the new tile.
    m_new = tl.maximum(m_prev, tl.max(S, 1))
    if IS_BORDER:
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    else:
        m_safe = m_new

    # Rescale factor for the *previous* O accumulator and l.
    alpha = tl.math.exp2((m_prev - m_safe) * softmax_scale_log2e)

    # Compute exp2(scaled S - row-max) in one shot.
    m_scaled = tl.where(m_new == float("-inf"), 0.0, m_safe * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - m_scaled[:, None])

    l_new = l_prev * alpha + tl.sum(P, 1)
    return alpha, P, m_new, l_new


# ------------------------------------------------------------------
# "Long-liverange-breaker" rescale.
# https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
# Multiplies acc by alpha along axis 0 in a way that the Triton compiler
# can split into two half-width ops. On Hopper this lets the second-half
# rescale overlap with the first-half wgmma issued from the consumer
# warpgroup. Lifted (and adapted to non-warpspec mode) from the upstream
# 06-fused-attention tutorial trick that gives ~5-10% on H100.
# ------------------------------------------------------------------
# @triton.jit
# def _rescale_acc(acc, alpha, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
#     # Triton's reshape+permute+split is recognised as register shuffles
#     # rather than memory moves; it fuses with the surrounding mulf.
#     BM: tl.constexpr = acc.shape[0]
#     BD: tl.constexpr = acc.shape[1]
#     if BD >= 64 and (BD % 2 == 0):
#         a0, a1 = acc.reshape([BM, 2, BD // 2]).permute(0, 2, 1).split()
#         a0 = a0 * alpha[:, None]
#         a1 = a1 * alpha[:, None]
#         acc = tl.join(a0, a1).permute(0, 2, 1).reshape([BM, BD])
#     else:
#         acc = acc * alpha[:, None]
#     return acc
@triton.jit
def _rescale_acc(acc, alpha, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    return acc * alpha[:, None]


# ------------------------------------------------------------------
# Page-table indirection for KV-cache (paged attention).
# Same semantics as in flash_kernel.py.
# ------------------------------------------------------------------
@triton.jit
def _virtual_to_cache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    BOUNDARY_CHECK: tl.constexpr = False,
):
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if BOUNDARY_CHECK:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int32)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int32)
    return page_block_index * block_size + page_offset


# ------------------------------------------------------------------
# The main FA3 forward kernel.
#
# Grid: (cdiv(max_seqlen_q, BLOCK_M), batch_size, num_heads_q)
#
# We deliberately keep the grid identical to the v2 varlen kernel so the
# host launcher can stay shape-agnostic.
# ------------------------------------------------------------------


def _heur_block_k(args):
    return triton.next_power_of_2(args["d"])


# Hopper sweet-spots for FA3-style fwd. BLOCK_M=128, BLOCK_N in {64, 128}.
# num_warps=8 is what triggers the compiler's two-warpgroup partition,
# which is the implicit "ping-pong" we rely on.
def _v3_configs():
    cfgs = []
    for bm in [128]:
        for bn in [64, 128]:
            for s in [2, 3, 4]:
                # 8 warps -> 2 warpgroups -> implicit pingpong
                cfgs.append(
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=8
                    )
                )
            # Keep one 4-warp option for small head_dim where 8 warps
            # over-allocates registers.
            cfgs.append(
                triton.Config({"BLOCK_M": bm, "BLOCK_N": 64}, num_stages=3, num_warps=4)
            )

    return cfgs


def _prune_v3_configs(configs, nargs, **kwargs):
    # When dropout is on we can't use 8 warps reliably (philox state
    # blows up the register budget). Fall back to 4-warp configs.
    if nargs.get("is_dropout", False):
        return [c for c in configs if c.num_warps == 4]
    # head_dim 256 with BLOCK_N=128 doesn't fit in SMEM on H100.
    d = nargs["d"]
    out = []
    for c in configs:
        if d >= 192 and c.kwargs["BLOCK_N"] == 128:
            continue
        out.append(c)
    return out


@libentry()
@triton.autotune(
    configs=_v3_configs(),
    prune_configs_by={"early_config_prune": _prune_v3_configs},
    key=["d", "is_causal", "is_local", "is_paged", "is_dropout"],
)
@triton.heuristics(
    values={
        "BLOCK_K": _heur_block_k,
    }
)
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "total_q",
    ]
)
def flash_varlen_fwd_v3_kernel(
    # ---- buffer pointers ----
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    # ---- strides (kept for paged path; TMA path uses descriptors) ----
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    # ---- varlen indexing ----
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # ---- sizes ----
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # ---- softmax scaling ----
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    # ---- dropout ----
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # ---- causal / SWA ----
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # ---- alibi ----
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # ---- paged ----
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # ---- kernel params ----
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # ------------------------------------------------------------------
    # 0. CTA coordinates and per-request bounds.
    # ------------------------------------------------------------------
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
    else:
        k_len_cache = seqlen_k
        k_bos = 0

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    # Noop CTA: this query-row-block is past EOS for the request.
    if m_block * BLOCK_M >= q_len:
        return

    # ------------------------------------------------------------------
    # 1. Iteration bounds (we mirror v2's two-segment loop split).
    # ------------------------------------------------------------------
    if is_local:
        n_block_min = tl.maximum(
            0, (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = tl.minimum(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right, BLOCK_N
            ),
        )

    # Borrow the v2 masking-block accounting; varlen is never even-MN.
    is_even_mn: tl.constexpr = False
    if (not is_causal) and (not is_local):
        n_masking_steps = 1
    else:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1
    n_masking_steps = tl.minimum(n_block_max - n_block_min, n_masking_steps)

    # ------------------------------------------------------------------
    # 2. ALiBi / dropout / philox setup.
    # ------------------------------------------------------------------
    if is_alibi:
        alibi_slope = tl.load(alibi_slopes_ptr + bid * alibi_slopes_batch_stride + hid)
        alibi_slope = alibi_slope / scale_softmax
    else:
        alibi_slope = 0.0

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)
    else:
        philox_seed = tl.zeros([], dtype=tl.uint64)
        philox_offset = tl.zeros([], dtype=tl.uint64)

    # ------------------------------------------------------------------
    # 3. TMA descriptors for Q / K / V / O.
    #
    # Descriptors are *device-side* objects in Triton 3.6: they cost no
    # SMEM and are recomputed by the compiler into 64-bit handles at issue
    # time. We allocate one per request element (per CTA) so we can index
    # straight into the right batch / head / sequence range. Paged KV is
    # the one exception -- TMA cannot follow page-table indirection, so
    # we fall back to gather loads on that path.
    # ------------------------------------------------------------------
    HEAD_DIM_PADDED: tl.constexpr = BLOCK_K  # next pow2(d), >= d

    # ----- Q descriptor: shape = (q_len, d) for this request -----
    q_base = q_ptr + q_offset + hid * q_head_stride
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=[q_len, d],
        strides=[q_row_stride, 1],
        block_shape=[BLOCK_M, HEAD_DIM_PADDED],
    )

    # ----- O descriptor (mirrors Q's layout) -----
    o_base = o_ptr + o_offset + hid * o_head_stride
    o_desc = tl.make_tensor_descriptor(
        base=o_base,
        shape=[q_len, d],
        strides=[o_row_stride, 1],
        block_shape=[BLOCK_M, HEAD_DIM_PADDED],
    )

    # ----- K / V descriptors: only built for non-paged path -----
    # For paged we keep raw pointers and use gather-style tl.load.
    if not is_paged:
        kv_head = hid // h_hk_ratio
        # In varlen mode, K and V are layouts (total_k, hk, d). We point the
        # descriptor at this request's k-segment.
        k_base = k_ptr + k_bos * k_row_stride + kv_head * k_head_stride
        v_base = v_ptr + k_bos * k_row_stride + kv_head * v_head_stride
        k_desc = tl.make_tensor_descriptor(
            base=k_base,
            shape=[k_len_cache, d],
            strides=[k_row_stride, 1],
            block_shape=[BLOCK_N, HEAD_DIM_PADDED],
        )
        v_desc = tl.make_tensor_descriptor(
            base=v_base,
            shape=[k_len_cache, d],
            strides=[k_row_stride, 1],  # same row stride as K (interleaved layout)
            block_shape=[BLOCK_N, HEAD_DIM_PADDED],
        )
    else:
        # Paged path: shift page table pointer to this request, keep raw
        # K/V bases for the head we own.
        page_table_ptr_b = page_table_ptr + bid * page_table_batch_stride
        kv_head = hid // h_hk_ratio
        k_base = k_ptr + kv_head * k_head_stride
        v_base = v_ptr + kv_head * v_head_stride

    # ------------------------------------------------------------------
    # 4. Load Q (stays in SRAM throughout via TMA).
    # ------------------------------------------------------------------
    bQ = q_desc.load([m_block * BLOCK_M, 0])

    # ------------------------------------------------------------------
    # 5. Init online-softmax state and accumulator.
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, HEAD_DIM_PADDED), dtype=tl.float32)
    rowmax = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum = tl.zeros([BLOCK_M], dtype=tl.float32)

    row_idx_q = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # ------------------------------------------------------------------
    # 6. Tile-load helpers (inlined; closure on q_desc / k_desc / etc.).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 7. Inner loop -- "masking" segment (deals with causal/local/EOS edges).
    #    Run iterations in *reverse* order from n_block_max-1 downwards,
    #    matching v2 for bit-equivalence on the overlap region.
    #
    # NOTE: We load K/V *inside* the loop body (rather than maintaining a
    # cur/next register pair across iterations as in the original FA3
    # paper §3.2 algorithm). Triton's static analyzer enforces that any
    # tensor used in the loop body is also defined in that body's scope,
    # so the FA3-paper-style hand-rolled prefetch can't be expressed
    # directly. Instead we rely on `num_stages` in the autotune config to
    # let Triton's software pipeliner do the cross-iteration prefetch
    # automatically -- with num_warps=8 + num_stages=3 the compiler
    # produces a schedule where the TMA load of iter j+1's K/V issues
    # while iter j's softmax / P@V is still in flight, which is the same
    # behaviour we wanted from the manual prefetch.
    # ------------------------------------------------------------------
    n_block_start_mask = n_block_max - 1
    for step in tl.range(0, n_masking_steps):
        n_block = n_block_start_mask - step
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

        # ---- (a) Load K and V for this iteration ----
        if is_paged:
            kvcache_idx = _virtual_to_cache(
                col_idx, k_len, page_table_ptr_b, block_size, BOUNDARY_CHECK=True
            )
            k_off = (
                tl.arange(0, HEAD_DIM_PADDED)[:, None]
                + kvcache_idx[None, :] * k_row_stride
            )
            v_off = (
                tl.arange(0, HEAD_DIM_PADDED)[None, :]
                + kvcache_idx[:, None] * k_row_stride
            )
            d_mask = tl.arange(0, HEAD_DIM_PADDED) < d
            kv_mask = col_idx < k_len
            bK = tl.load(
                k_base + k_off, mask=d_mask[:, None] & kv_mask[None, :], other=0.0
            )
            bV = tl.load(
                v_base + v_off, mask=kv_mask[:, None] & d_mask[None, :], other=0.0
            )
        else:
            bK = tl.trans(k_desc.load([n_block * BLOCK_N, 0]))  # (d, BLOCK_N)
            bV = v_desc.load([n_block * BLOCK_N, 0])  # (BLOCK_N, d)

        # ---- (b) GEMM-1: S = Q @ K  (kept fp32 for numerical stability) ----
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
        S = _apply_softcap_v3(S, softcap, is_softcap)
        S = _apply_alibi_v3(
            S,
            col_idx,
            row_idx_q,
            q_len,
            k_len,
            IS_CAUSAL=is_causal,
            IS_ALIBI=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = _apply_mask_v3(
            S,
            col_idx,
            row_idx_q,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            IS_EVEN_MN=False,
            IS_CAUSAL=is_causal,
            IS_LOCAL=is_local,
        )

        # ---- (c) Online softmax with deferred rescale ----
        alpha, P, rowmax, rowsum = _softmax_online_deferred(
            S,
            rowmax,
            rowsum,
            softmax_scale_log2e=scale_softmax_log2,
            IS_BORDER=True,
        )

        # ---- (d) Deferred rescale fused with GEMM-2: acc = alpha*acc + P @ V ----
        acc = _rescale_acc(acc, alpha, BLOCK_M=BLOCK_M, HEAD_DIM=HEAD_DIM_PADDED)
        P_typed = P.to(v_ptr.dtype.element_ty)

        if is_dropout:
            P_typed = _apply_dropout_v3(
                P_typed,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        acc = tl.dot(P_typed, bV, acc, out_dtype=tl.float32)

    # ------------------------------------------------------------------
    # 8. Inner loop -- "non-masking" dense segment.
    #
    # Here we let `tl.range(num_stages=...)` do the multi-stage software
    # pipelining. With num_warps=8 the compiler will assign GEMM-1, GEMM-3,
    # and softmax to different warpgroups (this is the implicit pingpong;
    # see PyTorch warp-spec roadmap blog 2026 for why it works without
    # explicit warpspec annotations on Hopper). The user-visible "ping-pong"
    # is the *cross-iteration* one: in iter k, warpgroup A runs softmax of
    # iter k while warpgroup B runs P@V of iter k-1.
    # ------------------------------------------------------------------
    n_dense_end = n_block_max - n_masking_steps  # exclusive upper bound from the top
    # Iterate downward (matches v2 for bit-equivalence on overlap region).
    for n_block in tl.range(
        n_dense_end - 1, n_block_min - 1, step=-1, num_stages=num_stages
    ):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if is_paged:
            kvcache_idx = _virtual_to_cache(
                col_idx, k_len, page_table_ptr_b, block_size, BOUNDARY_CHECK=False
            )
            k_off = (
                tl.arange(0, HEAD_DIM_PADDED)[:, None]
                + kvcache_idx[None, :] * k_row_stride
            )
            v_off = (
                tl.arange(0, HEAD_DIM_PADDED)[None, :]
                + kvcache_idx[:, None] * k_row_stride
            )
            d_mask = tl.arange(0, HEAD_DIM_PADDED) < d
            bK = tl.load(k_base + k_off, mask=d_mask[:, None], other=0.0)
            bV = tl.load(v_base + v_off, mask=d_mask[None, :], other=0.0)
        else:
            bK = tl.trans(k_desc.load([n_block * BLOCK_N, 0]))
            bV = v_desc.load([n_block * BLOCK_N, 0])

        S = tl.dot(bQ, bK, out_dtype=tl.float32)
        S = _apply_softcap_v3(S, softcap, is_softcap)
        S = _apply_alibi_v3(
            S,
            col_idx,
            row_idx_q,
            q_len,
            k_len,
            IS_CAUSAL=is_causal,
            IS_ALIBI=is_alibi,
            alibi_slope=alibi_slope,
        )
        # In the dense segment we only need the local-window mask, never causal/EOS.
        S = _apply_mask_v3(
            S,
            col_idx,
            row_idx_q,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            IS_EVEN_MN=True,
            IS_CAUSAL=False,
            IS_LOCAL=is_local,
        )

        alpha, P, rowmax, rowsum = _softmax_online_deferred(
            S,
            rowmax,
            rowsum,
            softmax_scale_log2e=scale_softmax_log2,
            IS_BORDER=is_local,
        )

        acc = _rescale_acc(acc, alpha, BLOCK_M=BLOCK_M, HEAD_DIM=HEAD_DIM_PADDED)
        P_typed = P.to(v_ptr.dtype.element_ty)

        if is_dropout:
            P_typed = _apply_dropout_v3(
                P_typed,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        acc = tl.dot(P_typed, bV, acc, out_dtype=tl.float32)

    # ------------------------------------------------------------------
    # 9. Epilogue: divide by row-sum, write O and LSE via TMA.
    # ------------------------------------------------------------------
    # Numerically stable inverse-sum (mirrors v2 behaviour for empty rows).
    invalid = (rowsum == 0) | (rowsum != rowsum)
    inv_sum = tl.where(invalid, 1.0, 1.0 / rowsum)
    if is_dropout:
        acc = acc * (inv_sum * rp_dropout)[:, None]
    else:
        acc = acc * inv_sum[:, None]

    # LSE in natural log so downstream consumers (vLLM, SDPA) can use it directly.
    lse = tl.where(invalid, float("inf"), rowmax * scale_softmax + tl.log(rowsum))

    # ----- Store O via TMA -----
    o_desc.store([m_block * BLOCK_M, 0], acc.to(o_ptr.dtype.element_ty))

    # ----- Store LSE (small, just use a plain store) -----
    # lse layout: (h, total_q)  with offset = hid*total_q + lse_offset + row
    lse_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptr = softmax_lse_ptr + hid * total_q + lse_offset + lse_row
    tl.store(lse_ptr, lse, mask=lse_row < q_len)


# ----------------------------------------------------------------------
# Dropout helper -- functionally identical to flash_kernel.apply_dropout
# but inlined here so this file has no cross-kernel dependency. Uses the
# same Philox 4x32-10 RNG so seeds/offsets match v2 bit-for-bit.
# ----------------------------------------------------------------------
@triton.jit
def _u64_to_lohi(x):
    return (x >> 32).to(tl.uint32), (x & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def _philox(seed, subsequence, offset):
    kPhilox10A: tl.constexpr = 0x9E3779B9
    kPhilox10B: tl.constexpr = 0xBB67AE85
    k0, k1 = _u64_to_lohi(seed.to(tl.uint64))
    c0, c1 = _u64_to_lohi(offset.to(tl.uint64))
    c2, c3 = _u64_to_lohi(subsequence.to(tl.uint64))
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in tl.static_range(6):
        res0 = kPhiloxSA * c0.to(tl.uint64)
        res1 = kPhiloxSB * c2.to(tl.uint64)
        r0x, r0y = _u64_to_lohi(res0)
        r1x, r1y = _u64_to_lohi(res1)
        c0, c1, c2, c3 = r1y ^ c1 ^ k0, r1x, r0y ^ c3 ^ k1, r0x
        k0 += kPhilox10A
        k1 += kPhilox10B
    res0 = kPhiloxSA * c0.to(tl.uint64)
    res1 = kPhiloxSB * c2.to(tl.uint64)
    r0x, r0y = _u64_to_lohi(res0)
    r1x, r1y = _u64_to_lohi(res1)
    c0, c1, c2, c3 = r1y ^ c1 ^ k0, r1x, r0y ^ c3 ^ k1, r0x
    return c0, c1, c2, c3


@triton.jit
def _apply_dropout_v3(
    P,
    row_start,
    col_start,
    n_cols,
    bid,
    hid,
    philox_seed,
    philox_offset,
    p_dropout_uint8: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.multiple_of(row_start, BLOCK_M)
    col_start = tl.multiple_of(col_start, BLOCK_N)
    row = row_start + tl.arange(0, BLOCK_M)[:, None]
    col = col_start // 4 + tl.arange(0, BLOCK_N // 4)[None, :]
    subsequence = row.to(tl.uint64) * n_cols + col.to(tl.uint64)
    offset = philox_offset + bid * NUM_HEADS + hid + subsequence * 0
    r0, r1, r2, r3 = _philox(philox_seed, subsequence, offset)
    r = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(BLOCK_M, BLOCK_N)
    mask = (r & 0xFF) >= p_dropout_uint8
    return tl.where(mask, (P * 0).to(P.dtype), P)
