import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _parse_rng_state(rng_state: torch.Tensor) -> Tuple[int, int]:
    rng_state = rng_state.cpu()
    return int(rng_state[0].item()), int(rng_state[1].item())


def _parse_philox(
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
) -> Tuple[int, int]:
    return int(philox_seed.item()), int(philox_offset.item())


def _smem_bytes(block_m, block_n, head_dim, num_stages, dtype_bytes=2):
    q_do_tile = 2 * block_m * head_dim * dtype_bytes
    kv_tile = 2 * block_n * head_dim * dtype_bytes * num_stages
    return q_do_tile + kv_tile


def _get_max_shared_mem():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(
        props,
        "shared_memory_per_block_optin",
        props.shared_memory_per_block,
    )


def _prune_configs(configs, named_args, **kwargs):
    head_dim = named_args.get("HEAD_DIM", 128)

    max_smem = _get_max_shared_mem() - 4 * 1024

    pruned = []
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        ns = cfg.num_stages
        if _smem_bytes(bm, bn, head_dim, ns) <= max_smem:
            pruned.append(cfg)

    if not pruned:
        pruned = [
            min(
                configs,
                key=lambda c: _smem_bytes(
                    c.kwargs["BLOCK_M"],
                    c.kwargs["BLOCK_N"],
                    head_dim,
                    c.num_stages,
                ),
            )
        ]
    return pruned


_DQ_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]

_DKV_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_DQ_CONFIGS,
    key=["seqlen_q", "seqlen_k", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _flash_attn_bwd_dq_fused(
    Q,
    K,
    V,
    dOut,
    Out,
    L,
    D,
    dQ,
    AttnBias,
    dBias,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    drop_stride_b,
    drop_stride_h,
    drop_stride_m,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_db,
    stride_dh,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DO_BIAS_GRAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_q_idx = tl.program_id(2)
    head_k_idx = head_q_idx // group_size

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    q_base = batch_idx * stride_qb + head_q_idx * stride_qh
    o_base = batch_idx * stride_ob + head_q_idx * stride_oh

    q = tl.load(
        Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    do = tl.load(
        dOut + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    o = tl.load(
        Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    Di = tl.sum(o * do.to(tl.float32), axis=1)

    tl.store(
        D + batch_idx * stride_db + head_q_idx * stride_dh + offs_m,
        Di,
        mask=mask_m,
    )

    Li = tl.load(
        L + batch_idx * stride_lb + head_q_idx * stride_lh + offs_m,
        mask=mask_m,
        other=0.0,
    )

    start_n_min = 0
    start_n_max = tl.cdiv(seqlen_k, BLOCK_N)
    if HAS_WINDOW:
        start_n_min = tl.maximum(
            start_n_min,
            (start_m * BLOCK_M - window_size_left) // BLOCK_N,
        )
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + window_size_right + BLOCK_N) // BLOCK_N,
        )
    if IS_CAUSAL:
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + BLOCK_N) // BLOCK_N,
        )

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes + head_q_idx)

    k_base = batch_idx * stride_kb + head_k_idx * stride_kh
    v_base = batch_idx * stride_vb + head_k_idx * stride_vh

    for start_n in range(start_n_min, start_n_max):
        offs_n_c = start_n * BLOCK_N + offs_n
        mask_n = offs_n_c < seqlen_k

        k = tl.load(
            K + k_base + offs_n_c[:, None] * stride_km + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V + v_base + offs_n_c[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bb = batch_idx * stride_bb + head_q_idx * stride_bh
            bias_block = tl.load(
                AttnBias
                + bb
                + offs_m[:, None] * stride_bm
                + offs_n_c[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m[:, None] - offs_n_c[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * drop_stride_b
                + head_q_idx * drop_stride_h
                + offs_m[:, None] * drop_stride_m
                + offs_n_c[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dq = dq + tl.dot(ds.to(q.dtype), k)

        if DO_BIAS_GRAD:
            ds_bias = p * (dp - Di[:, None])
            bb_out = batch_idx * stride_bb + head_q_idx * stride_bh
            tl.store(
                dBias
                + bb_out
                + offs_m[:, None] * stride_bm
                + offs_n_c[None, :] * stride_bn,
                ds_bias,
                mask=mask_m[:, None] & mask_n[None, :],
            )

    tl.store(
        dQ + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        dq.to(q.dtype),
        mask=mask_m[:, None],
    )


@triton.autotune(
    configs=_DKV_CONFIGS,
    key=["seqlen_q", "seqlen_k", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _flash_attn_bwd_dkv(
    Q,
    K,
    V,
    dOut,
    L,
    D,
    dK,
    dV,
    AttnBias,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    drop_stride_b,
    drop_stride_h,
    drop_stride_m,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_lb,
    stride_lh,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_n = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_k_idx = tl.program_id(2)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seqlen_k

    k_base = batch_idx * stride_kb + head_k_idx * stride_kh
    v_base = batch_idx * stride_vb + head_k_idx * stride_vh

    if IS_CAUSAL:
        if start_n * BLOCK_N >= seqlen_q:
            dk_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

            k_tmp = tl.load(
                K + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None],
                other=0.0,
            )
            v_tmp = tl.load(
                V + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None],
                other=0.0,
            )
            tl.store(
                dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                dk_zero.to(k_tmp.dtype),
                mask=mask_n[:, None],
            )
            tl.store(
                dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                dv_zero.to(v_tmp.dtype),
                mask=mask_n[:, None],
            )
            return

    k = tl.load(
        K + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    num_tiles_m = tl.cdiv(seqlen_q, BLOCK_M)
    global_m_min = 0
    global_m_max = num_tiles_m

    if HAS_WINDOW:
        global_m_max = tl.minimum(
            global_m_max,
            ((start_n + 1) * BLOCK_N - 1 + window_size_left + BLOCK_M) // BLOCK_M,
        )
        global_m_min = tl.maximum(
            global_m_min,
            (start_n * BLOCK_N - window_size_right) // BLOCK_M,
        )

    if IS_CAUSAL:
        global_m_min = tl.maximum(global_m_min, (start_n * BLOCK_N) // BLOCK_M)

    valid_tiles_m = global_m_max - global_m_min
    if valid_tiles_m <= 0:
        tl.store(
            dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
            dk.to(k.dtype),
            mask=mask_n[:, None],
        )
        tl.store(
            dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
            dv.to(v.dtype),
            mask=mask_n[:, None],
        )
        return

    total_iters = group_size * valid_tiles_m

    for linear in range(total_iters):
        gq_offset = linear // valid_tiles_m
        tile_m = global_m_min + linear % valid_tiles_m
        head_q_idx = head_k_idx * group_size + gq_offset

        q_base = batch_idx * stride_qb + head_q_idx * stride_qh
        o_base = batch_idx * stride_ob + head_q_idx * stride_oh
        ld_base = batch_idx * stride_lb + head_q_idx * stride_lh

        offs_m_c = tile_m * BLOCK_M + offs_m
        mask_m = offs_m_c < seqlen_q

        q = tl.load(
            Q + q_base + offs_m_c[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            dOut + o_base + offs_m_c[:, None] * stride_om + offs_d[None, :] * stride_od,
            mask=mask_m[:, None],
            other=0.0,
        )
        Li = tl.load(L + ld_base + offs_m_c, mask=mask_m, other=0.0)
        Di = tl.load(D + ld_base + offs_m_c, mask=mask_m, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bb = batch_idx * stride_bb + head_q_idx * stride_bh
            bias_block = tl.load(
                AttnBias
                + bb
                + offs_m_c[:, None] * stride_bm
                + offs_n[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m_c[:, None] - offs_n[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            alibi_slope = tl.load(alibi_slopes + head_q_idx)
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * drop_stride_b
                + head_q_idx * drop_stride_h
                + offs_m_c[:, None] * drop_stride_m
                + offs_n[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dk = dk + tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = dv + tl.dot(tl.trans(p.to(q.dtype)), do)

    tl.store(
        dK + k_base + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
        dk.to(k.dtype),
        mask=mask_n[:, None],
    )
    tl.store(
        dV + v_base + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
        dv.to(v.dtype),
        mask=mask_n[:, None],
    )


@triton.autotune(
    configs=_DQ_CONFIGS,
    key=["max_seqlen_q", "max_seqlen_k", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _flash_attn_bwd_varlen_dq_fused(
    Q,
    K,
    V,
    dOut,
    Out,
    L,
    D,
    dQ,
    AttnBias,
    dBias,
    cu_seq_q,
    cu_seq_k,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    max_seqlen_q,
    max_seqlen_k,
    H_q,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_bm,
    stride_bh,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DO_BIAS_GRAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_q_idx = tl.program_id(2)
    head_k_idx = head_q_idx // group_size

    start_q = tl.load(cu_seq_q + batch_idx)
    seqlen_q = tl.load(cu_seq_q + batch_idx + 1) - start_q
    start_k = tl.load(cu_seq_k + batch_idx)
    seqlen_k = tl.load(cu_seq_k + batch_idx + 1) - start_k
    if start_m * BLOCK_M >= seqlen_q:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q
    m_phys = start_q + offs_m

    q = tl.load(
        Q
        + m_phys[:, None] * stride_qm
        + head_q_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    do = tl.load(
        dOut
        + m_phys[:, None] * stride_om
        + head_q_idx * stride_oh
        + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    )
    o = tl.load(
        Out
        + m_phys[:, None] * stride_om
        + head_q_idx * stride_oh
        + offs_d[None, :] * stride_od,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    Di = tl.sum(o * do.to(tl.float32), axis=1)

    tl.store(D + m_phys * H_q + head_q_idx, Di, mask=mask_m)

    Li = tl.load(L + m_phys * H_q + head_q_idx, mask=mask_m, other=0.0)

    start_n_min = 0
    start_n_max = tl.cdiv(seqlen_k, BLOCK_N)
    if HAS_WINDOW:
        start_n_min = tl.maximum(
            start_n_min,
            (start_m * BLOCK_M - window_size_left) // BLOCK_N,
        )
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + window_size_right + BLOCK_N) // BLOCK_N,
        )
    if IS_CAUSAL:
        start_n_max = tl.minimum(
            start_n_max,
            ((start_m + 1) * BLOCK_M - 1 + BLOCK_N) // BLOCK_N,
        )

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if HAS_ALIBI:
        alibi_slope = tl.load(alibi_slopes + head_q_idx)

    for start_n in range(start_n_min, start_n_max):
        offs_n_c = start_n * BLOCK_N + offs_n
        mask_n = offs_n_c < seqlen_k
        n_phys = start_k + offs_n_c

        k = tl.load(
            K
            + n_phys[:, None] * stride_km
            + head_k_idx * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V
            + n_phys[:, None] * stride_vm
            + head_k_idx * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bias_block = tl.load(
                AttnBias
                + m_phys[:, None] * stride_bm
                + head_q_idx * stride_bh
                + offs_n_c[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m[:, None] - offs_n_c[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * (max_seqlen_q * max_seqlen_k)
                + head_q_idx * max_seqlen_k
                + offs_m[:, None] * max_seqlen_k
                + offs_n_c[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dq = dq + tl.dot(ds.to(q.dtype), k)

        if DO_BIAS_GRAD:
            ds_bias = p * (dp - Di[:, None])
            tl.store(
                dBias
                + m_phys[:, None] * stride_bm
                + head_q_idx * stride_bh
                + offs_n_c[None, :] * stride_bn,
                ds_bias,
                mask=mask_m[:, None] & mask_n[None, :],
            )

    tl.store(
        dQ
        + m_phys[:, None] * stride_qm
        + head_q_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        dq.to(q.dtype),
        mask=mask_m[:, None],
    )


@triton.autotune(
    configs=_DKV_CONFIGS,
    key=["max_seqlen_q", "max_seqlen_k", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _flash_attn_bwd_varlen_dkv(
    Q,
    K,
    V,
    dOut,
    L,
    D,
    dK,
    dV,
    AttnBias,
    cu_seq_q,
    cu_seq_k,
    alibi_slopes,
    sm_scale,
    group_size,
    window_size_left,
    window_size_right,
    dropout_p,
    drop_scale,
    seed,
    base_offset,
    max_seqlen_q,
    max_seqlen_k,
    H_q,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_bm,
    stride_bh,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_n = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_k_idx = tl.program_id(2)

    start_q = tl.load(cu_seq_q + batch_idx)
    seqlen_q = tl.load(cu_seq_q + batch_idx + 1) - start_q
    start_k = tl.load(cu_seq_k + batch_idx)
    seqlen_k = tl.load(cu_seq_k + batch_idx + 1) - start_k
    if start_n * BLOCK_N >= seqlen_k:
        return

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seqlen_k
    n_phys = start_k + offs_n

    if IS_CAUSAL:
        if start_n * BLOCK_N >= seqlen_q:
            dk_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_zero = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            k_tmp = tl.load(
                K
                + n_phys[:, None] * stride_km
                + head_k_idx * stride_kh
                + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None],
                other=0.0,
            )
            v_tmp = tl.load(
                V
                + n_phys[:, None] * stride_vm
                + head_k_idx * stride_vh
                + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None],
                other=0.0,
            )
            tl.store(
                dK
                + n_phys[:, None] * stride_km
                + head_k_idx * stride_kh
                + offs_d[None, :] * stride_kd,
                dk_zero.to(k_tmp.dtype),
                mask=mask_n[:, None],
            )
            tl.store(
                dV
                + n_phys[:, None] * stride_vm
                + head_k_idx * stride_vh
                + offs_d[None, :] * stride_vd,
                dv_zero.to(v_tmp.dtype),
                mask=mask_n[:, None],
            )
            return

    k = tl.load(
        K
        + n_phys[:, None] * stride_km
        + head_k_idx * stride_kh
        + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        V
        + n_phys[:, None] * stride_vm
        + head_k_idx * stride_vh
        + offs_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    num_tiles_m = tl.cdiv(seqlen_q, BLOCK_M)
    global_m_min = 0
    global_m_max = num_tiles_m

    if HAS_WINDOW:
        global_m_max = tl.minimum(
            global_m_max,
            ((start_n + 1) * BLOCK_N - 1 + window_size_left + BLOCK_M) // BLOCK_M,
        )
        global_m_min = tl.maximum(
            global_m_min,
            (start_n * BLOCK_N - window_size_right) // BLOCK_M,
        )

    if IS_CAUSAL:
        global_m_min = tl.maximum(global_m_min, (start_n * BLOCK_N) // BLOCK_M)

    valid_tiles_m = global_m_max - global_m_min
    if valid_tiles_m <= 0:
        tl.store(
            dK
            + n_phys[:, None] * stride_km
            + head_k_idx * stride_kh
            + offs_d[None, :] * stride_kd,
            dk.to(k.dtype),
            mask=mask_n[:, None],
        )
        tl.store(
            dV
            + n_phys[:, None] * stride_vm
            + head_k_idx * stride_vh
            + offs_d[None, :] * stride_vd,
            dv.to(v.dtype),
            mask=mask_n[:, None],
        )
        return

    total_iters = group_size * valid_tiles_m

    for linear in range(total_iters):
        gq_offset = linear // valid_tiles_m
        tile_m = global_m_min + linear % valid_tiles_m
        head_q_idx = head_k_idx * group_size + gq_offset

        offs_m_c = tile_m * BLOCK_M + offs_m
        mask_m = offs_m_c < seqlen_q
        m_phys = start_q + offs_m_c

        q = tl.load(
            Q
            + m_phys[:, None] * stride_qm
            + head_q_idx * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        do = tl.load(
            dOut
            + m_phys[:, None] * stride_om
            + head_q_idx * stride_oh
            + offs_d[None, :] * stride_od,
            mask=mask_m[:, None],
            other=0.0,
        )
        Di = tl.load(D + m_phys * H_q + head_q_idx, mask=mask_m, other=0.0)
        Li = tl.load(L + m_phys * H_q + head_q_idx, mask=mask_m, other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if HAS_BIAS:
            bias_block = tl.load(
                AttnBias
                + m_phys[:, None] * stride_bm
                + head_q_idx * stride_bh
                + offs_n[None, :] * stride_bn,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            s = s + bias_block.to(tl.float32)

        dist = offs_m_c[:, None] - offs_n[None, :]
        mask_s = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_s = mask_s & (dist >= 0)
        if HAS_WINDOW:
            mask_s = mask_s & (dist <= window_size_left) & (dist >= -window_size_right)
        if HAS_ALIBI:
            alibi_slope = tl.load(alibi_slopes + head_q_idx)
            s = s - alibi_slope * tl.abs(dist).to(tl.float32)

        s = tl.where(mask_s, s.to(tl.float32), float("-inf"))
        p = tl.exp(s - Li[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)

        if IS_DROPOUT:
            rng_off = (
                base_offset
                + batch_idx * (max_seqlen_q * max_seqlen_k)
                + head_q_idx * max_seqlen_k
                + offs_m_c[:, None] * max_seqlen_k
                + offs_n[None, :]
            )
            rand_vals = tl.rand(seed, rng_off)
            drop_mask = rand_vals > dropout_p
            dp = tl.where(drop_mask, dp * drop_scale, 0.0)
            p = tl.where(drop_mask, p * drop_scale, 0.0)

        ds = p * (dp - Di[:, None]) * sm_scale
        dk = dk + tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv = dv + tl.dot(tl.trans(p.to(q.dtype)), do)

    tl.store(
        dK
        + n_phys[:, None] * stride_km
        + head_k_idx * stride_kh
        + offs_d[None, :] * stride_kd,
        dk.to(k.dtype),
        mask=mask_n[:, None],
    )
    tl.store(
        dV
        + n_phys[:, None] * stride_vm
        + head_k_idx * stride_vh
        + offs_d[None, :] * stride_vd,
        dv.to(v.dtype),
        mask=mask_n[:, None],
    )


def _get_bias_strides_dense(bias):
    if bias is None:
        return 0, 0, 0, 0
    return bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3)


def _get_bias_strides_varlen(bias):
    if bias is None:
        return 0, 0, 0
    return bias.stride(0), bias.stride(1), bias.stride(2)


def flash_attn_backward(
    dOut,
    Q,
    K,
    V,
    Out,
    L,
    cu_seq_q=None,
    cu_seq_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    is_dropout=False,
    dropout_p=0.0,
    rng_state=None,
    is_causal=False,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    softmax_scale=None,
    attn_bias=None,
    bias_requires_grad=False,
):
    if is_dropout and dropout_p > 0.0 and rng_state is None:
        raise ValueError("开启 dropout 时，必须传入 rng_state")

    is_varlen = (cu_seq_q is not None) and (cu_seq_k is not None)

    if is_varlen:
        Total_Q, H_q, Head_Dim = Q.shape
        _, H_k, _ = K.shape
        Batch = cu_seq_q.shape[0] - 1
        SeqLen_q = max_seqlen_q
        SeqLen_k = max_seqlen_k
    else:
        Batch, SeqLen_q, H_q, Head_Dim = Q.shape
        _, SeqLen_k, H_k, _ = K.shape

    if H_q % H_k != 0:
        raise ValueError(f"H_q ({H_q}) 必须是 H_k ({H_k}) 的整数倍")
    group_size = H_q // H_k
    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Head_Dim))

    use_dropout = is_dropout and (dropout_p > 0.0)
    if use_dropout:
        seed_int, base_offset_int = rng_state
        drop_scale = 1.0 / (1.0 - dropout_p)
        drop_stride_b = H_q * SeqLen_q * SeqLen_k
        drop_stride_h = SeqLen_q * SeqLen_k
        drop_stride_m = SeqLen_k
    else:
        seed_int = base_offset_int = 0
        drop_scale = 1.0
        dropout_p = 0.0
        drop_stride_b = drop_stride_h = drop_stride_m = 0

    _inf_val = SeqLen_q + SeqLen_k + 1
    wl = _inf_val if window_size_left < 0 else window_size_left
    wr = _inf_val if window_size_right < 0 else window_size_right
    has_window = (wl < SeqLen_q) or (wr < SeqLen_k)
    has_alibi = alibi_slopes is not None
    alibi_ptr = alibi_slopes if has_alibi else Q
    has_bias = attn_bias is not None
    do_bias_grad = has_bias and bias_requires_grad

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    Out = Out.contiguous()
    dOut = dOut.contiguous()
    L = L.contiguous()
    if has_bias:
        attn_bias = attn_bias.contiguous()

    bias_ptr = attn_bias if has_bias else Q
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    dBias: Optional[torch.Tensor] = None
    if do_bias_grad:
        dBias = torch.empty_like(attn_bias)
    dbias_ptr = dBias if do_bias_grad else Q

    if is_varlen:
        D = torch.empty(Total_Q, H_q, device=Q.device, dtype=torch.float32)

        sb_m, sb_h, sb_n = _get_bias_strides_varlen(attn_bias if has_bias else None)

        grid_dq = lambda META: (triton.cdiv(SeqLen_q, META["BLOCK_M"]), Batch, H_q)
        _flash_attn_bwd_varlen_dq_fused[grid_dq](
            Q,
            K,
            V,
            dOut,
            Out,
            L,
            D,
            dQ,
            bias_ptr,
            dbias_ptr,
            cu_seq_q,
            cu_seq_k,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            max_seqlen_q,
            max_seqlen_k,
            H_q,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            sb_m,
            sb_h,
            sb_n,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
            DO_BIAS_GRAD=do_bias_grad,
        )

        grid_dkv = lambda META: (triton.cdiv(SeqLen_k, META["BLOCK_N"]), Batch, H_k)
        _flash_attn_bwd_varlen_dkv[grid_dkv](
            Q,
            K,
            V,
            dOut,
            L,
            D,
            dK,
            dV,
            bias_ptr,
            cu_seq_q,
            cu_seq_k,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            max_seqlen_q,
            max_seqlen_k,
            H_q,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            sb_m,
            sb_h,
            sb_n,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
        )

    else:
        assert L.shape == (
            Batch,
            H_q,
            SeqLen_q,
        ), f"L.shape {L.shape} != ({Batch}, {H_q}, {SeqLen_q})"
        D = torch.empty_like(L)

        sb_b, sb_h, sb_m, sb_n = _get_bias_strides_dense(
            attn_bias if has_bias else None
        )

        grid_dq = lambda META: (triton.cdiv(SeqLen_q, META["BLOCK_M"]), Batch, H_q)
        _flash_attn_bwd_dq_fused[grid_dq](
            Q,
            K,
            V,
            dOut,
            Out,
            L,
            D,
            dQ,
            bias_ptr,
            dbias_ptr,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            drop_stride_b,
            drop_stride_h,
            drop_stride_m,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            Out.stride(3),
            L.stride(0),
            L.stride(1),
            D.stride(0),
            D.stride(1),
            sb_b,
            sb_h,
            sb_m,
            sb_n,
            SeqLen_q,
            SeqLen_k,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
            DO_BIAS_GRAD=do_bias_grad,
        )

        grid_dkv = lambda META: (triton.cdiv(SeqLen_k, META["BLOCK_N"]), Batch, H_k)
        _flash_attn_bwd_dkv[grid_dkv](
            Q,
            K,
            V,
            dOut,
            L,
            D,
            dK,
            dV,
            bias_ptr,
            alibi_ptr,
            scale,
            group_size,
            wl,
            wr,
            dropout_p,
            drop_scale,
            seed_int,
            base_offset_int,
            drop_stride_b,
            drop_stride_h,
            drop_stride_m,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            K.stride(3),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            V.stride(3),
            Out.stride(0),
            Out.stride(1),
            Out.stride(2),
            Out.stride(3),
            L.stride(0),
            L.stride(1),
            sb_b,
            sb_h,
            sb_m,
            sb_n,
            SeqLen_q,
            SeqLen_k,
            HEAD_DIM=Head_Dim,
            IS_CAUSAL=is_causal,
            IS_DROPOUT=use_dropout,
            HAS_WINDOW=has_window,
            HAS_ALIBI=has_alibi,
            HAS_BIAS=has_bias,
        )

    return dQ, dK, dV, dBias


def flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    rng_state,
    unused,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
):
    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_rng_state(rng_state) if is_dropout else None
    wl = -1 if window_size_left is None else int(window_size_left)
    wr = -1 if window_size_right is None else int(window_size_right)
    dQ, dK, dV, _ = flash_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cu_seq_q=cum_seq_q,
        cu_seq_k=cum_seq_k,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        softmax_scale=scale,
    )
    return dQ, dK, dV


def scaled_dot_product_flash_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    philox_seed,
    philox_offset,
    *,
    scale=None,
):
    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    use_varlen = (cum_seq_q is not None) and (cum_seq_k is not None)
    dQ, dK, dV, _ = flash_attn_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        cu_seq_q=cum_seq_q if use_varlen else None,
        cu_seq_k=cum_seq_k if use_varlen else None,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
    )
    return dQ, dK, dV


def scaled_dot_product_cudnn_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    *,
    scale=None,
    bias_requires_grad=False,
):
    grad_out_bshd = grad_out.permute(0, 2, 1, 3).contiguous()
    query_bshd = query.permute(0, 2, 1, 3).contiguous()
    key_bshd = key.permute(0, 2, 1, 3).contiguous()
    value_bshd = value.permute(0, 2, 1, 3).contiguous()
    out_bshd = out.permute(0, 2, 1, 3).contiguous()
    lse = logsumexp.squeeze(-1).float()

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    use_varlen = (cum_seq_q is not None) and (cum_seq_k is not None)

    dQ_bshd, dK_bshd, dV_bshd, dBias = flash_attn_backward(
        grad_out_bshd,
        query_bshd,
        key_bshd,
        value_bshd,
        out_bshd,
        lse,
        cu_seq_q=cum_seq_q if use_varlen else None,
        cu_seq_k=cum_seq_k if use_varlen else None,
        max_seqlen_q=int(max_q),
        max_seqlen_k=int(max_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
        attn_bias=attn_bias,
        bias_requires_grad=bias_requires_grad,
    )

    dQ = dQ_bshd.permute(0, 2, 1, 3).contiguous()
    dK = dK_bshd.permute(0, 2, 1, 3).contiguous()
    dV = dV_bshd.permute(0, 2, 1, 3).contiguous()
    return dQ, dK, dV


def efficient_attention_backward(
    grad_out_,
    query,
    key,
    value,
    bias,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    logsumexp,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    *,
    scale=None,
    num_splits_key=None,
    window_size=None,
    shared_storage_dqdkdv=False,
):
    if custom_mask_type == 0:
        is_causal, wl, wr = False, -1, -1
    elif custom_mask_type == 1:
        is_causal, wl, wr = True, -1, -1
    elif custom_mask_type == 2:
        is_causal = False
        wl = 0
        wr = max_seqlen_q + max_seqlen_k + 1
    else:
        raise ValueError(f"未知 custom_mask_type: {custom_mask_type}")

    if window_size is not None and window_size >= 0:
        wl = wr = window_size

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    use_varlen = (cu_seqlens_q is not None) and (cu_seqlens_k is not None)
    return flash_attn_backward(
        grad_out_,
        query,
        key,
        value,
        out,
        logsumexp,
        cu_seq_q=cu_seqlens_q if use_varlen else None,
        cu_seq_k=cu_seqlens_k if use_varlen else None,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        softmax_scale=scale,
        attn_bias=bias,
        bias_requires_grad=bias_requires_grad,
    )


def scaled_dot_product_efficient_attention_backward(
    grad_out_,
    query,
    key,
    value,
    attn_bias,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask,
    is_causal=False,
    *,
    scale=None,
):
    need_dq, need_dk, need_dv, need_dbias = grad_input_mask

    grad_out_bshd = grad_out_.permute(0, 2, 1, 3).contiguous()
    query_bshd = query.permute(0, 2, 1, 3).contiguous()
    key_bshd = key.permute(0, 2, 1, 3).contiguous()
    value_bshd = value.permute(0, 2, 1, 3).contiguous()
    out_bshd = out.permute(0, 2, 1, 3).contiguous()

    is_dropout = dropout_p > 0.0
    rng_tuple = _parse_philox(philox_seed, philox_offset) if is_dropout else None
    bias_req_grad = need_dbias and (attn_bias is not None)

    dQ_bshd, dK_bshd, dV_bshd, dBias = flash_attn_backward(
        grad_out_bshd,
        query_bshd,
        key_bshd,
        value_bshd,
        out_bshd,
        logsumexp,
        cu_seq_q=None,
        cu_seq_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        is_dropout=is_dropout,
        dropout_p=dropout_p,
        rng_state=rng_tuple,
        is_causal=is_causal,
        softmax_scale=scale,
        attn_bias=attn_bias,
        bias_requires_grad=bias_req_grad,
    )

    dQ_bhsd = dQ_bshd.permute(0, 2, 1, 3).contiguous()
    dK_bhsd = dK_bshd.permute(0, 2, 1, 3).contiguous()
    dV_bhsd = dV_bshd.permute(0, 2, 1, 3).contiguous()

    if not need_dq:
        dQ_bhsd = torch.zeros_like(query)
    if not need_dk:
        dK_bhsd = torch.zeros_like(key)
    if not need_dv:
        dV_bhsd = torch.zeros_like(value)
    if not need_dbias:
        dBias = None

    return dQ_bhsd, dK_bhsd, dV_bhsd, dBias
