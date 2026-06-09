import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i_2d,
    m_i_2d,
    Q_block_ptr,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    q_desc = smt.descriptor_load(Q_block_ptr, (0, 0))
    q = smt.view(q_desc, (0, 0), (BLOCK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

    offs_m_4d = tl.reshape(offs_m, (num_m_tiles, 1, MICRO_M, 1))
    offs_n_4d = tl.reshape(offs_n, (1, num_n_tiles, 1, MICRO_N))

    causal_offset = KV_CTX - Q_CTX

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_desc = smt.descriptor_load(K_block_ptr, (0, 0))
        k = smt.view(k_desc, (0, 0), (BLOCK_N, BLOCK_SIZE_K), (MICRO_N, MICRO_K))
        trans_k = tl.permute(k, (1, 0, 3, 2))

        qk = smt.dot(q, trans_k) * qk_scale

        mask_n = (start_n + offs_n) < KV_CTX
        mask_n_4d = tl.reshape(mask_n, (1, num_n_tiles, 1, MICRO_N))
        qk = tl.where(mask_n_4d, qk, -1.0e6)

        if (STAGE == 2) or (STAGE == 4):
            mask_causal = (offs_m_4d + causal_offset) >= (start_n + offs_n_4d)
            mask = mask_causal & mask_n_4d
            qk = tl.where(mask, qk, -1.0e6)

        qk_max_3 = tl.max(qk, axis=3)
        m_ij_2d = tl.max(qk_max_3, axis=1)
        m_ij_2d = tl.maximum(m_i_2d, m_ij_2d)

        m_ij_bc = tl.reshape(m_ij_2d, (num_m_tiles, 1, MICRO_M, 1))
        qk = qk - m_ij_bc

        p = tl.math.exp(qk)
        p_sum_3 = tl.sum(p, axis=3)
        l_ij_2d = tl.sum(p_sum_3, axis=1)

        alpha_2d = tl.math.exp(m_i_2d - m_ij_2d)
        l_i_2d = l_i_2d * alpha_2d + l_ij_2d

        alpha_bc = tl.reshape(alpha_2d, (num_m_tiles, 1, MICRO_M, 1))
        acc = acc * alpha_bc

        v_desc = smt.descriptor_load(V_block_ptr, (0, 0))
        v = smt.view(v_desc, (0, 0), (BLOCK_N, BLOCK_SIZE_K), (MICRO_K, MICRO_N))

        p_cast = p.to(v.dtype)

        p_cast = smt.view(p_cast, (0, 0), (BLOCK_M, BLOCK_N), (MICRO_M, MICRO_K))

        acc += smt.dot(p_cast, v)

        m_i_2d = m_ij_2d
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    return acc, l_i_2d, m_i_2d


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("attention"),
    key=["Q_CTX", "KV_CTX", "HEAD_DIM", "GROUP_SIZE"],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    M,
    Out,
    acc_buffer,
    sm_scale,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    Z,
    H_Q,
    H_KV,
    GROUP_SIZE,
    Q_CTX,
    KV_CTX,
    HEAD_DIM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_ctas: tl.constexpr,
):
    NUM_BLOCKS_M = tl.cdiv(Q_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H_Q

    pid = tl.program_id(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_hz_idx = (pid + num_ctas * block_idx) // NUM_BLOCKS_M
        task_m_idx = (pid + num_ctas * block_idx) % NUM_BLOCKS_M

        off_z = task_hz_idx // H_Q
        off_hq = task_hz_idx % H_Q
        off_hkv = off_hq // GROUP_SIZE

        q_offset = off_z.to(tl.int64) * stride_qz + off_hq.to(tl.int64) * stride_qh
        k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
        v_offset = off_z.to(tl.int64) * stride_vz + off_hkv.to(tl.int64) * stride_vh
        o_offset = off_z.to(tl.int64) * stride_oz + off_hq.to(tl.int64) * stride_oh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(Q_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(KV_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_SIZE_K),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(KV_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_SIZE_K),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(Q_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_SIZE_K),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        # compute tile counts from META-provided BLOCK_*/MICRO_*
        num_m_tiles: tl.constexpr = BLOCK_M // MICRO_M
        num_k_tiles: tl.constexpr = BLOCK_SIZE_K // MICRO_N

        m_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) - float("inf")
        l_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) + 1.0
        acc_4d = tl.zeros(
            [num_m_tiles, num_k_tiles, MICRO_M, MICRO_N], dtype=tl.float32
        )

        if STAGE == 4:
            acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                acc_4d,
                l_i_2d,
                m_i_2d,
                Q_block_ptr,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                BLOCK_SIZE_K,
                BLOCK_N,
                4,
                offs_m,
                offs_n,
                Q_CTX,
                KV_CTX,
                MICRO_M,
                MICRO_K,
                MICRO_N,
                BLOCK_M // MICRO_M,
                BLOCK_N // MICRO_N,
                BLOCK_SIZE_K // MICRO_N,
            )
        else:
            if STAGE & 1:
                acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                    acc_4d,
                    l_i_2d,
                    m_i_2d,
                    Q_block_ptr,
                    K_block_ptr,
                    V_block_ptr,
                    task_m_idx,
                    sm_scale,
                    BLOCK_M,
                    BLOCK_SIZE_K,
                    BLOCK_N,
                    4 - STAGE,
                    offs_m,
                    offs_n,
                    Q_CTX,
                    KV_CTX,
                    MICRO_M,
                    MICRO_K,
                    MICRO_N,
                    BLOCK_M // MICRO_M,
                    BLOCK_N // MICRO_N,
                    BLOCK_SIZE_K // MICRO_N,
                )
            if STAGE & 2:
                acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                    acc_4d,
                    l_i_2d,
                    m_i_2d,
                    Q_block_ptr,
                    K_block_ptr,
                    V_block_ptr,
                    task_m_idx,
                    sm_scale,
                    BLOCK_M,
                    BLOCK_SIZE_K,
                    BLOCK_N,
                    2,
                    offs_m,
                    offs_n,
                    Q_CTX,
                    KV_CTX,
                    MICRO_M,
                    MICRO_K,
                    MICRO_N,
                    BLOCK_M // MICRO_M,
                    BLOCK_N // MICRO_N,
                    BLOCK_SIZE_K // MICRO_N,
                )

        acc_2d = smt.view(acc_4d, (0, 0), (BLOCK_M, BLOCK_SIZE_K), (1, 1))
        m_i = tl.reshape(m_i_2d, (BLOCK_M,))
        l_i = tl.reshape(l_i_2d, (BLOCK_M,))

        m_i = m_i + tl.math.log(l_i)
        accumulator = acc_2d / l_i[:, None]

        mask_m = offs_m < Q_CTX
        m_ptrs = M + task_hz_idx * Q_CTX + offs_m
        tl.store(m_ptrs, m_i.to(M.type.element_ty), mask=mask_m)

        tl.store(
            O_block_ptr, accumulator.to(Out.type.element_ty), boundary_check=(0, 1)
        )


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal, enable_gqa: bool):
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K == v.shape[-1]
        BLOCK_SIZE_K = triton.next_power_of_2(HEAD_DIM_K)

        Q_CTX = q.shape[2]
        KV_CTX = k.shape[2]

        H_Q = q.shape[1]
        H_KV = k.shape[1]

        if H_Q != H_KV:
            enable_gqa = True

        if enable_gqa:
            assert (
                H_Q % H_KV == 0
            ), f"GQA requires H_Q % H_KV == 0, got H_Q={H_Q}, H_KV={H_KV}"
            GROUP_SIZE = H_Q // H_KV
        else:
            assert H_Q == H_KV
            GROUP_SIZE = 1

        o = torch.empty_like(q)
        acc = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
            dtype=torch.float32,
            device=q.device,
        )
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device
        )

        if is_causal:
            STAGE = 3 if (Q_CTX == KV_CTX) else 4
        else:
            STAGE = 1

        num_ctas = 16
        grid = lambda META: (META.get("num_ctas", num_ctas),)

        _attn_fwd[grid](
            q,
            k,
            v,
            M,
            o,
            acc,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            H_Q=H_Q,
            H_KV=H_KV,
            GROUP_SIZE=GROUP_SIZE,
            Q_CTX=Q_CTX,
            KV_CTX=KV_CTX,
            HEAD_DIM=HEAD_DIM_K,
            STAGE=STAGE,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        return o


def flash_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    return Attention.apply(query, key, value, scale, is_causal, enable_gqa)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    query = query.clone().contiguous()
    key = key.clone().contiguous()
    value = value.clone().contiguous()
    return Attention.apply(query, key, value, scale, is_causal, enable_gqa)
