"""Multi-head / multi-batch / GQA version of warp-specialized flash attention (H800 sm_90).

Extended from the single-head prototype in gluon_attn.py:
- Grid expanded from (cdiv(S,BLOCK_M),) to (cdiv(S,BLOCK_M), B*H_Q), second dimension indexes (batch, q_head).
- Descriptors all use 2D: flatten [B,H,S,D] with head/batch offset added to row coordinate.
  * Q/O : [B*H_Q*S, D], row offset qh*S
  * V   : [B*H_KV*S, D], row offset kvh*S
  * K^T : [B*H_KV*D, S], row offset kvh*D
- GQA: kvh = b*H_KV + (h_in // ratio), ratio = H_Q // H_KV.

Keep gluon_attn.py untouched.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

LOG2E = 1.4426950408889634


@gluon.jit
def _kv_producer(
    k_desc,
    v_desc,
    k_empty,
    k_ready,
    k_bufs,
    v_bufs,
    n_kv_blocks,
    k_row0,
    v_row0,
    BLOCK_N: gl.constexpr,
):
    """Producer: TMA loads K^T([D,BLOCK_N]) and V([BLOCK_N,D]) into smem multi-buffers.
    k_row0 / v_row0 are the row base addresses for the current head in the flattened tensor.
    """
    nbuf: gl.constexpr = k_bufs.type.shape[0]
    for i in range(n_kv_blocks):
        idx = i % nbuf
        phase = i // nbuf & 1
        e_bar = k_empty.index(idx)
        r_bar = k_ready.index(idx)
        mbarrier.wait(e_bar, phase ^ 1)
        koff = i * BLOCK_N
        mbarrier.expect(r_bar, k_desc.block_type.nbytes + v_desc.block_type.nbytes)
        # K^T tensor [B*H_KV*D, S]: columns are kv sequence dimension, row base is k_row0
        tma.async_copy_global_to_shared(
            k_desc, [k_row0, koff], r_bar, k_bufs.index(idx)
        )
        # V tensor [B*H_KV*S, D]: row = v_row0 + koff
        tma.async_copy_global_to_shared(
            v_desc, [v_row0 + koff, 0], r_bar, v_bufs.index(idx)
        )


@gluon.jit
def _attn_consumer(
    q_desc,
    o_desc,
    lse_ptr,
    k_empty,
    k_ready,
    k_bufs,
    v_bufs,
    n_kv_blocks,
    off_m,
    q_row0,
    o_row0,
    lse_row0,
    qk_scale,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    D: gl.constexpr,
    CAUSAL: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Consumer (default warpgroup): QK^T -> online softmax(exp2) -> P@V."""
    nbuf: gl.constexpr = k_bufs.type.shape[0]
    m_i_shape: gl.constexpr = 16
    k_dim: gl.constexpr = 16
    wpc: gl.constexpr = [num_warps, 1]

    qk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=wpc, instr_shape=[m_i_shape, BLOCK_N, k_dim]
    )
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=wpc, instr_shape=[m_i_shape, D, k_dim]
    )
    q_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_layout, k_width=2
    )
    p_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_layout, k_width=2
    )
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    # row_pv is reserved for potential future use in alternative layouts
    # row_pv: gl.constexpr = gl.SliceLayout(1, pv_layout)

    q_smem = gl.allocate_shared_memory(
        q_desc.dtype, q_desc.block_type.shape, q_desc.layout
    )
    q_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(q_bar, count=1)
    mbarrier.expect(q_bar, q_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(q_desc, [q_row0 + off_m, 0], q_bar, q_smem)
    mbarrier.wait(q_bar, 0)
    mbarrier.invalidate(q_bar)
    q = q_smem.load(q_reg_layout)

    acc = gl.zeros((BLOCK_M, D), dtype=gl.float32, layout=pv_layout)
    m_i = gl.full((BLOCK_M,), -float("inf"), gl.float32, layout=row_qk)
    l_i = gl.zeros((BLOCK_M,), dtype=gl.float32, layout=row_qk)
    offs_m = off_m + gl.arange(0, BLOCK_M, layout=row_qk)

    for i in range(n_kv_blocks):
        idx = i % nbuf
        phase = i // nbuf & 1
        e_bar = k_empty.index(idx)
        r_bar = k_ready.index(idx)
        mbarrier.wait(r_bar, phase ^ 1)

        kt = k_bufs.index(idx).load(gl.DotOperandLayout(1, qk_layout, k_width=2))
        qk = (
            warpgroup_mma(q, kt, acc=None, layout=qk_layout).to(
                dtype=gl.float32, layout=qk_layout
            )
            * qk_scale
        )

        if CAUSAL:
            offs_n = i * BLOCK_N + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, qk_layout)
            )
            qk = gl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        warpgroup_mma_wait(1)
        m_curr = gl.max(qk, axis=1)
        m_new = gl.maximum(m_i, m_curr)
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        m_i = m_new

        v = v_bufs.index(idx).load(gl.DotOperandLayout(1, pv_layout, k_width=2))
        p_reg = p.to(dtype=q.type.scalar, layout=p_reg_layout)
        acc = warpgroup_mma(p_reg, v, acc=acc, layout=pv_layout)
        warpgroup_mma_wait(1)
        mbarrier.arrive(e_bar)

    out = (acc / l_i[:, None]).to(dtype=q.type.scalar, layout=pv_layout)

    o_smem = gl.allocate_shared_memory(
        o_desc.dtype, o_desc.block_type.shape, o_desc.layout
    )
    o_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(o_bar, count=1)
    o_smem.store(out)
    fence_async_shared(True)
    mbarrier.arrive(o_bar)
    mbarrier.wait(o_bar, 0)
    tma.async_copy_shared_to_global(o_desc, o_smem, [o_row0 + off_m, 0])
    tma.async_copy_commit_group()
    tma.async_copy_wait_group(0)

    # Store LSE: log-sum-exp = m_i / LOG2E + log(l_i).
    # Use explicit BlockedLayout for 1D pointer store.
    lse_out = (m_i / LOG2E + gl.log(l_i)).to(
        dtype=gl.float32, layout=gl.BlockedLayout([BLOCK_M], [1])
    )
    offs_lse = lse_row0 + off_m.to(layout=gl.BlockedLayout([BLOCK_M], [1]))
    lse_ptr.store(offs_lse, lse_out, mask=offs_lse < lse_ptr.size)


@gluon.jit
def attn_ws_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    S,
    qk_scale,
    H_Q,
    H_KV,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    D: gl.constexpr,
    CAUSAL: gl.constexpr,
    NBUF: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Warp-specialized kernel: producer/consumer + multi-buffer pipeline.

    Handles flattened tensors:
      Q: [B*H_Q*S, D]
      K^T: [B*H_KV*D, S]
      V: [B*H_KV*S, D]
      O: [B*H_Q*S, D]
      LSE: [B*H_Q*S] (1D pointer, float32)
    """
    # Grid: (cdiv(S, BLOCK_M), B*H_Q)
    off_m = gl.program_id(0) * BLOCK_M
    qh = gl.program_id(1)
    if off_m >= S:
        return

    # GQA index: qh = b*H_Q + h_in → kvh = b*H_KV + (h_in // ratio)
    ratio = H_Q // H_KV
    b = qh // H_Q
    h_in = qh % H_Q
    kvh = b * H_KV + (h_in // ratio)

    # Row base addresses in flattened tensors
    q_row0 = qh * S
    o_row0 = q_row0
    k_row0 = kvh * D
    v_row0 = kvh * S
    lse_row0 = qh * S

    n_kv_blocks = triton.cdiv(S, BLOCK_N)
    k_bufs = gl.allocate_shared_memory(
        k_desc.dtype, [NBUF, *k_desc.block_type.shape], k_desc.layout
    )
    v_bufs = gl.allocate_shared_memory(
        v_desc.dtype, [NBUF, *v_desc.block_type.shape], v_desc.layout
    )
    k_empty = gl.allocate_shared_memory(gl.int64, [NBUF], mbarrier.MBarrierLayout())
    k_ready = gl.allocate_shared_memory(gl.int64, [NBUF], mbarrier.MBarrierLayout())

    for i in range(NBUF):
        mbarrier.init(k_empty.index(i), count=1)
        mbarrier.init(k_ready.index(i), count=0)
        mbarrier.arrive(k_empty.index(i))

    kv_producer = gluon.producer(
        _kv_producer,
        inputs=[
            k_desc,
            v_desc,
            k_empty,
            k_ready,
            k_bufs,
            v_bufs,
            n_kv_blocks,
            k_row0,
            v_row0,
            BLOCK_N,
        ],
        barrier=None,
    )

    gluon.warpgroup(_attn_consumer, size=num_warps)(
        q_desc,
        o_desc,
        lse_ptr,
        k_empty,
        k_ready,
        k_bufs,
        v_bufs,
        n_kv_blocks,
        off_m,
        q_row0,
        o_row0,
        lse_row0,
        qk_scale,
        BLOCK_M,
        BLOCK_N,
        D,
        CAUSAL,
        num_warps,
    )
    kv_producer.join()


def _prepare(q, k, v, BLOCK_M=128, BLOCK_N=128, D=None):
    """Prepare descriptors and LSE buffer for the kernel.
    Input: q,k,v [B, S, H, D] (non-contiguous from transpose is fine).
    Returns a context dict."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, S, H_Q, d = q.shape
    H_KV = k.shape[2]
    D = D or d
    if D != d:
        raise ValueError(f"D={D} != head_dim={d}")

    # Flatten to 2D
    q_flat = q.view(B * H_Q * S, D)
    v_flat = v.view(B * H_KV * S, D)
    k_t_flat = k.transpose(2, 3).contiguous().view(B * H_KV * D, S)

    q_desc = TensorDescriptor(q_flat, [[BLOCK_M, D], [1, 1], [0, 0]])
    k_desc = TensorDescriptor(k_t_flat, [[D, BLOCK_N], [1, 1], [0, 0]])
    v_desc = TensorDescriptor(v_flat, [[BLOCK_N, D], [1, 1], [0, 0]])
    o_flat = torch.empty_like(q_flat)
    o_desc = TensorDescriptor(o_flat, [[BLOCK_M, D], [1, 1], [0, 0]])

    # LSE: [B, H_Q, S] float32
    lse_flat = torch.empty(B * H_Q * S, dtype=torch.float32, device=q.device)

    return {
        "q": q_desc,
        "k": k_desc,
        "v": v_desc,
        "o": o_desc,
        "lsef": lse_flat,
        "shape": (B, S, H_Q, H_KV, D),
    }


def _run_kernel(ctx, causal, BLOCK_M=128, BLOCK_N=128, NBUF=3, num_warps=4):
    """Run the kernel in-place, update ctx['o'] and ctx['lsef']."""
    B, S, H_Q, H_KV, D = ctx["shape"]
    lse_ptr = ctx["lsef"]
    scale = ctx.get("softmax_scale") or (1.0 / (D**0.5))
    qk_scale = scale * LOG2E
    grid = (triton.cdiv(S, BLOCK_M), B * H_Q)
    attn_ws_kernel[grid](
        ctx["q"],
        ctx["k"],
        ctx["v"],
        ctx["o"],
        lse_ptr,
        S,
        qk_scale,
        H_Q,
        H_KV,
        BLOCK_M,
        BLOCK_N,
        D,
        causal,
        NBUF,
        num_warps=num_warps,
        maxnreg=128,
    )


def attn_ws_mha(
    q,
    k,
    v,
    causal=True,
    BLOCK_M=128,
    BLOCK_N=128,
    NBUF=3,
    num_warps=4,
    return_lse=False,
    softmax_scale=None,
):
    """End-to-end: input/output are [B, S, H, D] (matching production kernel).
    return_lse=True returns additional LSE [B, H_Q, S] float32."""
    ctx = _prepare(q, k, v, BLOCK_M, BLOCK_N, q.shape[-1])
    ctx["softmax_scale"] = softmax_scale
    _run_kernel(ctx, causal, BLOCK_M, BLOCK_N, NBUF, num_warps)
    B, S, H_Q, H_KV, D = ctx["shape"]
    out = ctx["o"].tensor.view(B, H_Q, S, D).transpose(1, 2).contiguous()
    if return_lse:
        lse = ctx["lsef"].view(B, H_Q, S)
        return out, lse
    return out
