import math
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.runtime.backend._spacemit.ops.attn_vfwmacc import _attn_m1_fused_host
from flag_gems.utils import libentry, libtuner


@triton.jit
def _attn_fwd_inner(
    acc,
    l_acc,
    m_i_2d,
    Q_block_ptr,
    K_block_ptr,
    V_block_ptr,
    ONES_block_ptr,
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
    # Pass 2 of the two-pass flash: m_i_2d already holds the true row max
    # over this program's whole attended range (from _attn_max_pass), so the
    # KV loop carries no reductions — softmax is exp(qk - m_bc) with a
    # loop-invariant broadcast, and the row sums come from a second engine
    # dot against a ones matrix (l_acc) instead of tl.sum. tl.max/tl.sum on
    # the 4D engine-layout tile force a layout round-trip through scratch
    # memory that used to cost ~85% of the kernel's wall time.
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
    ones_desc = smt.descriptor_load(ONES_block_ptr, (0, 0))
    ones_v = smt.view(ones_desc, (0, 0), (BLOCK_N, MICRO_N), (MICRO_K, MICRO_N))
    m_bc = tl.reshape(m_i_2d, (num_m_tiles, 1, MICRO_M, 1))
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
        p = tl.math.exp(qk - m_bc)
        v_desc = smt.descriptor_load(V_block_ptr, (0, 0))
        v = smt.view(v_desc, (0, 0), (BLOCK_N, BLOCK_SIZE_K), (MICRO_K, MICRO_N))
        p_cast = p.to(v.dtype)
        p_cast = smt.view(p_cast, (0, 0), (BLOCK_M, BLOCK_N), (MICRO_M, MICRO_K))
        acc += smt.dot(p_cast, v)
        l_acc += smt.dot(p_cast, ones_v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_acc


@triton.jit
def _attn_max_pass(
    m_acc_4d,
    Q_block_ptr,
    K_block_ptr,
    qk_scale,
    lo,
    hi,
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    Q_CTX: tl.constexpr,
    KV_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
):
    # Pass 1 of the two-pass flash: row max over the attended KV range.
    # Loads K only (never V). Accumulates the running max elementwise in
    # engine layout and reduces once after the loop: a per-block tl.max
    # forces the same layout round trip through scratch memory that the
    # two-pass removed from pass 2.
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
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
        if CAUSAL:
            mask_causal = (offs_m_4d + causal_offset) >= (start_n + offs_n_4d)
            mask = mask_causal & mask_n_4d
            qk = tl.where(mask, qk, -1.0e6)
        m_acc_4d = tl.maximum(m_acc_4d, qk)
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    qk_max_3 = tl.max(m_acc_4d, axis=3)
    m_row = tl.max(qk_max_3, axis=1)
    return m_row


def _spacemit_attention_configs():
    # Tuned configs for "attention" are shared across CPU backends and may
    # carry options this kernel does not take (PRE_LOAD_V) plus a num_ctas
    # tuned for another flavor; strip the former and pin num_ctas to the
    # launcher default (grid size = number of persistent programs).
    cfgs = runtime.get_tuned_config("attention")
    # BLOCK_M sets the smt scratch arena size; the TCM pool is 256KB per
    # worker and the tuned BLOCK_M=128 requests ~485KB, which tcm_malloc
    # fails with NULL (the kernel then segfaults on its first scratch init
    # store). Clamp to a size that fits the pool.
    block_m = int(os.environ.get("ATTN_FLASH_BLOCK_M", "32"))
    block_n = int(os.environ.get("ATTN_FLASH_BLOCK_N", "32"))
    for c in cfgs:
        c.kwargs.pop("PRE_LOAD_V", None)
        c.num_ctas = 16
        c.kwargs["BLOCK_M"] = block_m
        c.kwargs["BLOCK_N"] = block_n
    return cfgs


@libentry()
@libtuner(
    configs=_spacemit_attention_configs(),
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
    ONES,
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

        ONES_block_ptr = tl.make_block_ptr(
            base=ONES,
            shape=(BLOCK_N, MICRO_N),
            strides=(MICRO_N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, MICRO_N),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        # compute tile counts from META-provided BLOCK_*/MICRO_*
        num_m_tiles: tl.constexpr = BLOCK_M // MICRO_M
        num_k_tiles: tl.constexpr = BLOCK_SIZE_K // MICRO_N

        m_acc_4d = tl.zeros(
            [num_m_tiles, BLOCK_N // MICRO_N, MICRO_M, MICRO_N], dtype=tl.float32
        ) - float("inf")
        l_acc_4d = tl.zeros([num_m_tiles, 1, MICRO_M, MICRO_N], dtype=tl.float32)
        acc_4d = tl.zeros(
            [num_m_tiles, num_k_tiles, MICRO_M, MICRO_N], dtype=tl.float32
        )

        # Pass 1: row max over the program's whole attended range. The
        # per-STAGE ranges match the union of the pass-2 loop ranges below
        # (STAGE 3's two loops partition [0, (start_m+1)*BLOCK_M)).
        if STAGE == 3:
            lo_m, hi_m = 0, (task_m_idx + 1) * BLOCK_M
        else:
            lo_m, hi_m = 0, KV_CTX
        m_i_2d = _attn_max_pass(
            m_acc_4d,
            Q_block_ptr,
            K_block_ptr,
            sm_scale,
            lo_m,
            hi_m,
            STAGE != 1,
            offs_m,
            offs_n,
            Q_CTX,
            KV_CTX,
            BLOCK_M,
            BLOCK_SIZE_K,
            BLOCK_N,
            MICRO_M,
            MICRO_K,
            MICRO_N,
            BLOCK_M // MICRO_M,
            BLOCK_N // MICRO_N,
        )

        if STAGE == 4:
            acc_4d, l_acc_4d = _attn_fwd_inner(
                acc_4d,
                l_acc_4d,
                m_i_2d,
                Q_block_ptr,
                K_block_ptr,
                V_block_ptr,
                ONES_block_ptr,
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
            )
        else:
            if STAGE & 1:
                acc_4d, l_acc_4d = _attn_fwd_inner(
                    acc_4d,
                    l_acc_4d,
                    m_i_2d,
                    Q_block_ptr,
                    K_block_ptr,
                    V_block_ptr,
                    ONES_block_ptr,
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
                )
            if STAGE & 2:
                acc_4d, l_acc_4d = _attn_fwd_inner(
                    acc_4d,
                    l_acc_4d,
                    m_i_2d,
                    Q_block_ptr,
                    K_block_ptr,
                    V_block_ptr,
                    ONES_block_ptr,
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
                )

        acc_2d = smt.view(acc_4d, (0, 0), (BLOCK_M, BLOCK_SIZE_K), (1, 1))
        # Collapse the ones-dot replicas in 4D, mirroring the m-pass reduces
        # on qk: every MICRO_N column of l_acc holds the same row sum, so the
        # max is exact. Reducing a 2D view of the loop-carried tile instead
        # emits unpack+transpose, which folds into a perm'd spestruct.unpack
        # that SpeStructToVector's LowerUnpackPattern cannot lower (assert
        # !perm_inner && !perm_outer; repro: probe_artifacts/flash_crash_dump).
        l_max_3 = tl.max(l_acc_4d, axis=3)
        l_row = tl.max(l_max_3, axis=1)
        l_i = tl.reshape(l_row, (BLOCK_M,))
        m_i = tl.reshape(m_i_2d, (BLOCK_M,)) + tl.math.log(l_i)
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

        # Ones matrix for the engine-side row-sum dot in _attn_fwd_inner.
        # Width 32 must stay tied to MICRO_N=32 in the launch below; BLOCK_N
        # must match the clamp in _spacemit_attention_configs (every config
        # is forced to it, so the tuner cannot pick anything else).
        block_n = int(os.environ.get("ATTN_FLASH_BLOCK_N", "32"))
        ones = torch.ones((block_n, 32), dtype=v.dtype, device=q.device)
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
            ones,
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
            MICRO_M=16,
            MICRO_K=8,
            MICRO_N=32,
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


def _sdpa_m1_fused(q, k, v, scale, is_causal):
    """Fused M=1 flash attention for decode — single kernel launch.

    QKT + softmax + AV fused via default-path spine_raw (vexp/vmacc/vreduce).
    grid=(Hq,), each program handles one Q head (hq/hk computed in JIT wrapper
    via tl.program_id). Causal is a no-op for M=1 (query at position Mk-1
    attends to all keys 0..Mk-1). Validated: max_diff=4.88e-04, 0.354ms/call.
    """
    B, Hq, Mq, D = q.shape
    _, Hkv, Mk, _ = k.shape
    assert Mq == 1 and B == 1
    n_rep = Hq // Hkv
    out = torch.empty((B, Hq, Mq, D), dtype=q.dtype, device=q.device)
    scratch = torch.empty((Hq, Mk), dtype=torch.float32, device=q.device)
    scale_t = torch.tensor([scale], dtype=torch.float32, device=q.device)
    _attn_m1_fused_host[(Hq,)](
        q.view(Hq, D),
        k.view(Hkv, Mk, D),
        v.view(Hkv, Mk, D),
        scratch.view(Hq * Mk),
        scale_t,
        out.view(Hq, D),
        Mk,
        D,
        n_rep,
        Hq,
    )
    return out


def _sdpa_python_fallback(query, key, value, attn_mask, dropout_p, is_causal, scale):
    """Manual QK^T + softmax + AV for shapes/dtypes the fused kernels reject.

    Must not call aten scaled_dot_product_attention (recursion once this op is
    registered as an aten override). Computes in fp32 because the backend has
    no bf16 dot and cannot pack head_dim=16 dots.
    """
    assert dropout_p == 0.0, "fallback only supports dropout_p=0.0"
    B, Hq, Mq, D = query.shape
    _, Hkv, Mk, _ = key.shape
    if Hq != Hkv:
        n_rep = Hq // Hkv
        key = key.unsqueeze(2).expand(B, Hkv, n_rep, Mk, D).reshape(B, Hq, Mk, D)
        value = value.unsqueeze(2).expand(B, Hkv, n_rep, Mk, D).reshape(B, Hq, Mk, D)
    attn = query.float() @ key.float().transpose(-1, -2) * scale
    if is_causal:
        q_idx = torch.arange(Mq, device=query.device).unsqueeze(-1)
        k_idx = torch.arange(Mk, device=query.device).unsqueeze(0)
        attn = attn.masked_fill(k_idx - q_idx > Mk - Mq, float("-inf"))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.nn.functional.softmax(attn, dim=-1)
    out = attn @ value.float()
    return out.to(query.dtype)


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
    # Fast path: K3 attention kernels (f16, B=1, GQA; smt.dot flash for Mq>1)
    B, Hq, Mq, D = query.shape
    _, Hkv, Mk, _ = key.shape
    if (
        query.dtype == torch.float16
        and D in (64, 128)
        and B == 1
        and Hq % Hkv == 0
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and attn_mask is None
        and dropout_p == 0.0
    ):
        sc = scale if scale is not None else (1.0 / math.sqrt(D))
        if Mq == 1 and D == 128:
            return _sdpa_m1_fused(query, key, value, sc, is_causal)
        return flash_attention(
            query, key, value, is_causal=is_causal, scale=sc, enable_gqa=enable_gqa
        )
    sc = scale if scale is not None else (1.0 / math.sqrt(D))
    return _sdpa_python_fallback(query, key, value, attn_mask, dropout_p, is_causal, sc)


def scaled_dot_product_flash_attention_for_cpu(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    *,
    attn_mask=None,
    scale=None,
):
    """Bridge: aten::_scaled_dot_product_flash_attention_for_cpu → FlagGems SDPA.

    Inductor decomposes `aten::scaled_dot_product_attention` (CompositeImplicitAutograd)
    down to `aten::_scaled_dot_product_flash_attention_for_cpu` during FakeTensor tracing,
    bypassing FlagGems' CPU registration. Registering this wrapper on the decomposed op
    routes Inductor-compiled graphs back through FlagGems SDPA.

    Native op returns (output, attn_matrix); we return (output, None) since FlagGems
    SDPA does not compute the attention matrix.

    The 4D attn_mask passed by HF's sdpa_attention_forward is a standard causal mask
    (0 on/below diagonal, -inf above) for Qwen3 prefill without padding, and all-zeros
    for decode (Mq=1). We convert: Mq==1 → is_causal=False; Mq>1 → is_causal=True.
    """
    Mq = query.shape[2]
    if attn_mask is not None:
        is_causal = Mq > 1
        attn_mask = None
    out = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    return (out, None)
