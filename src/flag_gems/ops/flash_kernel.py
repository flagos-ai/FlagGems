import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, tl_extra_shim


@triton.jit
def u64_to_lohi(x):
    return (x >> 32).to(tl.uint32), (x & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def u64_from_lohi(lo, hi):
    return hi.to(tl.uint64) << 32 + lo.to(tl.uint64)


@triton.jit
def philox_(seed, subsequence, offset):
    kPhilox10A: tl.constexpr = 0x9E3779B9
    kPhilox10B: tl.constexpr = 0xBB67AE85
    k0, k1 = u64_to_lohi(seed.to(tl.uint64))
    c0, c1 = u64_to_lohi(offset.to(tl.uint64))
    c2, c3 = u64_to_lohi(subsequence.to(tl.uint64))

    # pragma unroll
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in tl.static_range(6):
        res0 = kPhiloxSA * c0.to(tl.uint64)
        res1 = kPhiloxSB * c2.to(tl.uint64)
        res0_x, res0_y = u64_to_lohi(res0)
        res1_x, res1_y = u64_to_lohi(res1)
        c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x
        k0 += kPhilox10A
        k1 += kPhilox10B

    res0 = kPhiloxSA * c0.to(tl.uint64)
    res1 = kPhiloxSB * c2.to(tl.uint64)
    res0_x, res0_y = u64_to_lohi(res0)
    res1_x, res1_y = u64_to_lohi(res1)
    c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x

    return c0, c1, c2, c3


@triton.jit
def apply_dropout_mask(
    P,
    mask,
    encode_dropout_in_sign_bit: tl.constexpr,
):
    if encode_dropout_in_sign_bit:
        P = tl.where(mask, -P, P)
    else:
        P = tl.where(mask, (P * 0).to(P.dtype), P)
    return P


@triton.jit
def apply_dropout(
    P,
    row_start,
    col_start,
    n_cols,
    bid,
    hid,
    philox_seed,
    philox_offset,
    p_dropout_uint8: tl.constexpr,
    is_dropout: tl.constexpr,
    encode_dropout_in_sign_bit: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if is_dropout:
        row_start = tl.multiple_of(row_start, BLOCK_M)
        col_start = tl.multiple_of(col_start, BLOCK_N)
        row = row_start + tl.arange(0, BLOCK_M)[:, None]
        # Down scale col_idx by 4
        col = col_start // 4 + tl.arange(0, BLOCK_N // 4)[None, :]

        subsequence = row.to(tl.uint64) * n_cols + col.to(tl.uint64)

        offset = philox_offset + bid * NUM_HEADS + hid
        offset += subsequence * 0
        r0, r1, r2, r3 = philox_(philox_seed, subsequence, offset)

        r = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(BLOCK_M, BLOCK_N)

        mask = (r & 0xFF) >= p_dropout_uint8

        P = apply_dropout_mask(
            P, mask, encode_dropout_in_sign_bit=encode_dropout_in_sign_bit
        )
    return P


@triton.jit
def apply_alibi(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    is_causal: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slope: tl.constexpr = None,
):
    if is_alibi:
        if is_causal:
            # The row independent alibi bias renders the same attention output
            # as with the standard alibi because softmax is shift invariant, i.e.,
            # softmax(A + bias + const) = softamx(A + bias). The following two
            # biases are no different if causal is true.
            # bias_1 = [
            #   -4, -3, -2,  X, X,
            #   -4, -3, -2, -1, X,
            #   -4, -3, -2, -1, 0,
            # ]
            # bias_2 = [
            #   -2, -1, 0,  X,  X,
            #   -3, -2, -1, 0,  X,
            #   -4, -3, -2, -1, 0,
            # ]
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias

    return S


@triton.jit
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
):
    need_mask = is_causal | is_local | (not is_even_mn)
    # need_mask: tl.constexpr = is_causal | is_local
    if need_mask:
        # Extra care should be taken to void one-off errors: both col_lb and col_rb are inclusive!
        col_lb = max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = min(
            max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + window_size_right
        )

        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)

        if is_local:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )

        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)

    return S


@triton.jit
def softmax_rescale(
    O_acc,
    S,
    row_max,
    row_sum,
    softmax_scale_log2e: tl.constexpr,
    is_border: tl.constexpr,
    # is_init: tl.constexpr
):
    prev_max = row_max
    row_max = tl.maximum(row_max, tl.max(S, 1))

    if is_border:
        cur_max = tl.where(row_max == float("-inf"), 0, row_max)
    else:
        cur_max = row_max

    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]

    max_scaled = tl.where(row_max == float("-inf"), 0, row_max * softmax_scale_log2e)
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


@triton.jit
def apply_softcap(S, softcap, is_softcap: tl.constexpr):
    if is_softcap:
        S = tl_extra_shim.tanh(S * softcap)

    return S


def block_m_splitkv_heuristic(headdim):
    return 128 if headdim <= 128 else 64


def block_n_splitkv_heuristic(headdim):
    return 64 if headdim <= 64 else 32


def is_even_mn(M, N, BM, BN, WL, WR):
    if M % BM == 0 and N % BN == 0:
        if M % N == 0 or N % M == 0:
            if (WL == -1 or WL % BN == 0) and (WR == -1 or WR % BN == 0):
                return True
    return False


def block_m_splitkv_heuristic_spec_args(args):
    return 128 if args["d"] <= 128 else 64


def block_n_splitkv_heuristic_spec_args(args):
    return 64 if args["d"] <= 64 else 32


def is_even_mn_spec_args(args):
    if (
        args["seqlen_q"] % args["BLOCK_M"] == 0
        and args["seqlen_k"] % args["BLOCK_N"] == 0
    ):
        if (
            args["seqlen_q"] % args["seqlen_k"] == 0
            or args["seqlen_k"] % args["seqlen_q"] == 0
        ):
            if (
                args["window_size_left"] == -1
                or args["window_size_left"] % args["BLOCK_N"] == 0
            ) and (
                args["window_size_right"] == -1
                or args["window_size_right"] % args["BLOCK_N"] == 0
            ):
                return True
    return False


def keep(cfg, must_keep=None):
    BM = cfg.kwargs["BLOCK_M"]
    BN = cfg.kwargs["BLOCK_N"]
    w = cfg.num_warps

    # we always keep configurations in `must_keep`
    return (BM, BN, w) in ((128, 32, 4), (128, 128, 8)) or (
        must_keep and cfg in must_keep
    )


def prune_fwd_configs(configs, nargs, **kwargs):
    is_dropout = nargs["is_dropout"]
    if is_dropout:
        return list(
            filter(lambda cfg: cfg.num_warps == 4 and cfg.num_stages < 4, configs)
        )
    else:
        return configs


def flash_fwd_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.autotune(
    configs=list(filter(keep, runtime.get_tuned_config("attention"))),
    prune_configs_by={"early_config_prune": prune_fwd_configs},
    key=["d", "is_dropout"],
)
@triton.heuristics(
    values={
        "BLOCK_K": flash_fwd_kernel_heur_block_k,
        "PRE_LOAD_V": lambda args: False,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["window_size_left"],
            args["window_size_right"],
        ),
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
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
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k,
    cu_seqlens_k_ptr,
    is_seqused_k,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q: tl.constexpr,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    hid = bh % h
    bid = bh // h
    num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    # We draw a minimum covering frame on the attention map that this CTA is assigned to process.
    # The frame edges are rounded to multiples of BLOCK_M and BLOCK_N for rows and columns respectively.

    col_min = 0
    if is_local:
        col_min = max(0, m_block * BLOCK_M + seqlen_k - seqlen_q - window_size_left)
        if not IS_EVEN_MN:
            # round left
            col_min = (col_min // BLOCK_N) * BLOCK_N

    col_max = seqlen_k
    if is_causal or is_local:
        col_max += (m_block - num_m_blocks + 1) * BLOCK_M
        if is_local:
            col_max += window_size_right
        col_max = min(seqlen_k, col_max)

    if not IS_EVEN_MN:
        # round right
        col_max = tl.cdiv(col_max, BLOCK_N) * BLOCK_N

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            masking_cols: tl.constexpr = 0
        else:
            masking_cols: tl.constexpr = BLOCK_N
    elif (
        is_causal | is_local
    ) and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_cols: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) * BLOCK_N
    else:
        # local
        masking_cols: tl.constexpr = (tl.cdiv(BLOCK_M, BLOCK_N) + 1) * BLOCK_N

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    q_batch_stride = tl.multiple_of(q_batch_stride, d * h)
    q_ptr += bid * q_batch_stride + hid * q_head_stride
    row_start = m_block * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & d == BLOCK_K:
        Q = tl.load(q_ptr + q_off, cache_modifier=".cg")
    else:
        Q = tl.load(q_ptr + q_off, mask=qmask, cache_modifier=".cg")

    if return_softmax:
        p_ptr += (
            (bid * h + hid) * seqlen_q_rounded + m_block * BLOCK_M
        ) * seqlen_k_rounded
        p_offset = tl.arange(0, BLOCK_M)[:, None] * seqlen_k_rounded + tl.arange(
            0, BLOCK_N
        )
        p_bp0 = p_ptr + p_offset

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_batch_stride = tl.multiple_of(k_batch_stride, d * hk)
    h_hk_ratio = h // hk
    k_ptr += bid * k_batch_stride
    k_ptr += (hid // h_hk_ratio) * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += (hid // h_hk_ratio) * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    v_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_row_stride + tl.arange(0, BLOCK_K)[None, :]
    )

    p_bk0 = k_ptr + k_offset
    p_bv0 = v_ptr + v_offset

    if is_causal | is_local | (not IS_EVEN_MN):
        # Cut short masking cols if there's not enough cols out there
        masking_cols = min(col_max - col_min, masking_cols)
        for col_shift in tl.range(0, masking_cols, step=BLOCK_N):
            col_start = col_max - col_shift - BLOCK_N
            col_start = tl.multiple_of(col_start, BLOCK_N)
            off = col_start * k_row_stride
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_bk0 + off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
            elif d == BLOCK_K:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(p_bk0 + off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
            else:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_bk0 + off,
                    mask=kvmask[None, :] & dmask[:, None],
                    cache_modifier=".cg",
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & dmask[None, :],
                        cache_modifier=".cg",
                    )
            S = tl.dot(Q, K, allow_tf32=False)
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = col_start + tl.arange(0, BLOCK_N)
            row_idx = row_start + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            # tl.store(p_bp0 + col_start, S)
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=is_local,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
            )
            P = P.to(v_ptr.type.element_ty)

            if is_dropout:
                if return_softmax:
                    P_drop = P

                    P_drop = apply_dropout(
                        P_drop,
                        row_start,
                        col_start,
                        seqlen_k,
                        bid,
                        hid,
                        philox_seed,
                        philox_offset,
                        p_dropout_in_uint8_t,
                        is_dropout,
                        encode_dropout_in_sign_bit=True,
                        NUM_HEADS=h,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                    )
                    if IS_EVEN_MN:
                        tl.store(p_bp0 + col_start, P_drop)
                    else:
                        kvmask = col_idx < seqlen_k
                        tl.store(
                            p_bp0 + col_start, P_drop, mask=qmask & kvmask[None, :]
                        )

                P = apply_dropout(
                    P,
                    row_start,
                    col_start,
                    seqlen_k,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=False,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )

            if not PRE_LOAD_V:
                off = col_start * k_row_stride
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
                else:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & dmask[None, :],
                        cache_modifier=".cg",
                    )
            acc_ = tl.dot(P, V, acc_, allow_tf32=False)

    for col_start in tl.range(
        col_min, col_max - masking_cols, step=BLOCK_N, num_stages=num_stages
    ):
        col_start = tl.multiple_of(col_start, BLOCK_N)
        off = col_start * k_row_stride
        if d == BLOCK_K:
            K = tl.load(p_bk0 + off, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
        else:
            K = tl.load(p_bk0 + off, mask=dmask[:, None], cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, mask=dmask[None, :], cache_modifier=".cg")

        S = tl.dot(Q, K)
        S = apply_softcap(S, softcap, is_softcap)
        col_idx = col_start + tl.arange(0, BLOCK_N)
        row_idx = row_start + tl.arange(0, BLOCK_M)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
        )
        P = P.to(v_ptr.type.element_ty)

        if is_dropout:
            if return_softmax:
                P_drop = P
                P_drop = apply_dropout(
                    P_drop,
                    row_start,
                    col_start,
                    seqlen_k,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=True,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                if IS_EVEN_MN:
                    tl.store(p_bp0 + col_start, P_drop)
                else:
                    kvmask = col_idx < seqlen_k
                    tl.store(p_bp0 + col_start, P_drop, mask=qmask & kvmask[None, :])

            P = apply_dropout(
                P,
                row_start,
                col_start,
                seqlen_k,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        if not PRE_LOAD_V:
            off = col_start * k_row_stride
            if d == BLOCK_K:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
            else:
                V = tl.load(p_bv0 + off, mask=dmask[None, :], cache_modifier=".cg")
        acc_ = tl.dot(P, V, acc_)

    # LSE
    # Note, rowsum = exp(-rowmax) * exp(lse), therefore rowmax + log(rowsum) cancels
    # the effect of rowmax and outputs lse only.
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    if is_dropout:
        acc_ *= inv_sum[:, None] * rp_dropout
    else:
        acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_batch_stride = tl.multiple_of(o_batch_stride, d * h)
    o_ptr += bid * o_batch_stride
    o_ptr += hid * o_head_stride
    o_offset = row_idx[:, None] * o_row_stride + tl.arange(0, BLOCK_K)

    if IS_EVEN_MN & d == BLOCK_K:
        tl.store(o_ptr + o_offset, out)
    else:
        tl.store(o_ptr + o_offset, out, mask=qmask)

    # Write back lse
    p_lse = softmax_lse_ptr + (bid * h + hid) * seqlen_q
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    if IS_EVEN_MN:
        tl.store(p_lse + row_idx, lse)
    else:
        tl.store(p_lse + row_idx, lse, mask=row_idx < seqlen_q)


@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k"])
def flash_fwd_bh_parallel_kernel():
    # (TODO)
    pass


def flash_fwd_splitkv_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": block_m_splitkv_heuristic_spec_args,
        "BLOCK_N": block_n_splitkv_heuristic_spec_args,
        "BLOCK_K": flash_fwd_splitkv_kernel_heur_block_k,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
        "PRE_LOAD_V": lambda args: True,
        "IS_EVEN_MN": is_even_mn_spec_args,
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
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
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    blocks_per_split: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    split_id = tl.program_id(1)
    bid = tl.program_id(2) // h
    hid = tl.program_id(2) % h

    split_block_min = split_id * blocks_per_split
    split_block_max = split_block_min + blocks_per_split

    n_block_max = tl.cdiv(seqlen_k, BLOCK_N)
    if is_causal:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + window_size_right,
                BLOCK_N,
            ),
        )

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0

    if not is_causal:
        if IS_EVEN_MN:
            masking_block_min = n_block_max
        else:
            masking_block_min = n_block_max - 1
    elif is_causal and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N) - 1

    q_ptr += bid * q_batch_stride
    q_ptr += hid * q_head_stride
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    p_qm = q_ptr + q_off
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & BLOCK_K == d:
        Q = tl.load(p_qm, cache_modifier=".cg")
    else:
        Q = tl.load(p_qm, mask=qmask, cache_modifier=".cg")

    h_hk_ratio = h // hk
    k_ptr += bid * k_batch_stride
    k_ptr += (hid // h_hk_ratio) * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += (hid // h_hk_ratio) * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    p_k0 = k_ptr + k_offset

    v_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_row_stride + tl.arange(0, BLOCK_K)[None, :]
    )
    p_v0 = v_ptr + v_offset

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if split_block_max <= masking_block_min:
        # no masking needed
        for n_block in tl.range(
            split_block_min, split_block_max, num_stages=num_stages
        ):
            kv_off = n_block * BLOCK_N * k_row_stride
            if d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
            else:
                K = tl.load(
                    p_k0 + kv_off, mask=dmask[:, None], cache_modifier=".cg", other=0.0
                )
            if PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            S = tl.dot(Q, K)
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=False,
            )

            if not PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            P = P.to(v_ptr.type.element_ty)
            acc_ = tl.dot(P, V, acc_)
    else:
        for n_block in tl.range(split_block_min, min(split_block_max, n_block_max)):
            kv_off = n_block * BLOCK_N * k_row_stride
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            elif d == BLOCK_K:
                kvmask = col_idx < seqlen_k
                K = tl.load(p_k0 + kv_off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
            else:
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_k0 + kv_off,
                    mask=dmask[:, None] & kvmask[None, :],
                    cache_modifier=".cg",
                    other=0.0,
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )

            S = tl.dot(Q, K)
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=False,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
            )

            if not PRE_LOAD_V:
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            P = P.to(v_ptr.type.element_ty)
            acc_ = tl.dot(P, V, acc_)

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("-inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    # Rescale output
    acc_ *= inv_sum[:, None]

    # Write back output
    # o_splits layout = (n_splits, batch_size, num_heads, seqlen_q, head_size)
    # grid = (seq_block, split, batch * head)
    o_split_ptr = o_ptr
    # + split, batch, head offsets, seq_block offsets are already added in row_idx
    o_split_ptr += (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q * d
    o_split_offset = row_idx[:, None] * d + tl.arange(0, BLOCK_K)
    o_split_ptr = tl.multiple_of(o_split_ptr, d)
    p_om = o_split_ptr + o_split_offset

    if IS_EVEN_MN & BLOCK_K == d:
        tl.store(p_om, acc_, cache_modifier=".cg")
    else:
        tl.store(p_om, acc_, mask=qmask, cache_modifier=".cg")

    # Write back lse
    # lse_splits layout = (n_splits, batch_size, num_heads, seqlen_q)
    lse_split_ptr = softmax_lse_ptr
    # + split, batch, head, seq_block offsets
    lse_split_ptr += (
        split_id * tl.num_programs(2) + tl.program_id(2)
    ) * seqlen_q + m_block * BLOCK_M

    if IS_EVEN_MN:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse, cache_modifier=".cg")
    else:
        tl.store(
            lse_split_ptr + tl.arange(0, BLOCK_M),
            lse,
            mask=row_idx < seqlen_q,
            cache_modifier=".cg",
        )


@libentry()
@triton.jit
def flash_fwd_splitkv_combine_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    head_size: tl.constexpr,
    out_split_stride,
    lse_split_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    n_splits,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    q_total,
    MAX_N_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    lse_splits_ptr += pid * BLOCK_M
    lse_ptr += pid * BLOCK_M
    out_splits_ptr += pid * BLOCK_M * head_size
    out_ptr += pid * BLOCK_M * head_size

    # Subtracting maximum from each of the split lse's for better numerical stability
    lse_split_offset = (
        tl.arange(0, BLOCK_M)[:, None]
        + tl.arange(0, MAX_N_SPLITS)[None, :] * lse_split_stride
    )
    lse_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < q_total) & (
        tl.arange(0, MAX_N_SPLITS)[None, :] < n_splits
    )
    lse_splits = tl.load(
        lse_splits_ptr + lse_split_offset, mask=lse_split_mask, other=float("-inf")
    )
    max_lse = tl.max(lse_splits, 1)

    # Sum exp(lse(i) - max_lse) over all split i to obtain Z=sumexp(QK) up to a scaled factor exp(-max_lse)
    Zi_scaled = tl.exp(lse_splits - max_lse[:, None])
    Z_scaled = tl.sum(Zi_scaled, 1)
    Zi_Z = Zi_scaled / Z_scaled[:, None]

    # Write back LSE
    lse = tl.log(Z_scaled) + max_lse
    out_mask = pid * BLOCK_M + tl.arange(0, BLOCK_M) < q_total
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=out_mask)

    out_split_offset = (
        tl.arange(0, BLOCK_M)[:, None, None] * head_size
        + tl.arange(0, MAX_N_SPLITS)[None, :, None] * out_split_stride
        + tl.arange(0, BLOCK_K)[None, None, :]
    )
    out_split_mask = (
        (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None] < q_total)
        & (tl.arange(0, MAX_N_SPLITS)[None, :, None] < n_splits)
        & (tl.arange(0, BLOCK_K)[None, None, :] < head_size)
    )
    out_splits = tl.load(
        out_splits_ptr + out_split_offset, mask=out_split_mask, other=0.0
    )
    out = tl.sum(Zi_Z[:, :, None] * out_splits, 1)
    out = out.to(out_ptr.type.element_ty)

    # Write back output
    out_offset = tl.arange(0, BLOCK_M)[:, None] * out_s_stride + tl.arange(0, BLOCK_K)
    dmask = tl.arange(0, BLOCK_K) < head_size
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None] & dmask[None, :])


@triton.jit
def virtual_to_cache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    boundary_check: tl.constexpr = False,
):
    # virtual_index is the kv sequence index in the current batch element
    # page_table_ptr is already pointed at current batch element's block table entry
    # block_size is the size of each block in the page table
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if boundary_check:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int32)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int32)
    return page_block_index * block_size + page_offset


@triton.jit
def load_from_kvcache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    k_ptr_base,
    v_ptr_base,
    block_size,
    d: tl.constexpr,
    k_row_stride,
    BLOCK_K: tl.constexpr,
    boundary_check: tl.constexpr = False,
):
    kvcache_idx = virtual_to_cache(
        virtual_index, max_virtual_index, page_table_ptr, block_size, boundary_check
    )
    k_offset = tl.arange(0, BLOCK_K)[:, None] + kvcache_idx[None, :] * k_row_stride
    v_offset = tl.arange(0, BLOCK_K)[None, :] + kvcache_idx[:, None] * k_row_stride
    if d == BLOCK_K:
        bK = tl.load(k_ptr_base + k_offset)
        bV = tl.load(v_ptr_base + v_offset)
    else:
        bK = tl.load(
            k_ptr_base + k_offset, mask=tl.arange(0, BLOCK_K)[:, None] < d, other=0.0
        )
        bV = tl.load(
            v_ptr_base + v_offset, mask=tl.arange(0, BLOCK_K)[None, :] < d, other=0.0
        )
    return bK, bV


@libentry()
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
def flash_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
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
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
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
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    # num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        # Current request's start offset in the batched Q
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos * 1
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
        # k_offset = k_bos * k_row_stride
    else:
        k_len_cache = seqlen_k
        # k_offset = bid * k_batch_stride

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    # Noop CTA
    if m_block * BLOCK_M > q_len:
        return

    # is_even_mn = (q_len % BLOCK_M == 0) and (k_len % BLOCK_N == 0)
    is_even_mn: tl.constexpr = False

    if is_local:
        n_block_min = max(
            0, (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right, BLOCK_N
            ),
        )

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    # Locate the page table entry for the current batch element
    page_table_ptr += bid * page_table_batch_stride
    # Calculate the starting offset of q for the current head
    q_row_offset = hid * q_head_stride
    # Calculate the starting offset of k and v for the current head
    k_row_offset = (hid // h_hk_ratio) * k_head_stride
    # Shift the k, v pointers to align with the current head
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + k_row_offset

    gQ = tl.make_block_ptr(
        base=q_ptr + q_offset + q_row_offset,
        shape=(q_len, d),
        strides=(q_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )
    bQ = tl.load(gQ.advance([m_block * BLOCK_M, 0]), boundary_check=(0, 1))

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    if not is_causal and not is_local:
        n_masking_steps = 1
    elif is_even_mn:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1

    n_masking_steps = min(n_block_max - n_block_min, n_masking_steps)

    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_block = n_block_max - 1
    for step in tl.range(0, n_masking_steps):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        bK, bV = load_from_kvcache(
            col_idx,
            k_len,
            page_table_ptr,
            k_ptr_base,
            v_ptr_base,
            block_size,
            d,
            k_row_stride,
            BLOCK_K=BLOCK_K,
            boundary_check=True,
        )
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=is_even_mn,
            is_causal=is_causal,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=True,
        )
        P = P.to(v_ptr.type.element_ty)

        if is_dropout:
            P = apply_dropout(
                P,
                n_block * BLOCK_N,
                m_block * BLOCK_M,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )

        acc_ = tl.dot(P, bV, acc_)
        n_block -= 1

    for n_block in tl.range(
        n_block_max - n_masking_steps - 1, n_block_min - 1, step=-1
    ):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        bK, bV = load_from_kvcache(
            col_idx,
            k_len,
            page_table_ptr,
            k_ptr_base,
            v_ptr_base,
            block_size,
            d,
            k_row_stride,
            BLOCK_K=BLOCK_K,
        )
        S = tl.dot(bQ, bK, out_dtype=tl.float32)
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
        )
        P = P.to(v_ptr.type.element_ty)

        if is_dropout:
            P = apply_dropout(
                P,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        acc_ = tl.dot(P, bV, acc_)

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_row_offset = hid * o_head_stride

    gO = tl.make_block_ptr(
        base=o_ptr + o_offset + o_row_offset,
        shape=(q_len, d),
        strides=(o_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1),
    )
    tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))

    # Write back lse
    # lse shape: [h, total_q]
    softmax_lse_ptr += hid * total_q
    lse_row_offset = lse_offset + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=lse_row_offset < (lse_offset + q_len),
    )
