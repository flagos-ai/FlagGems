"""多头 / 多 batch / GQA 版 warp-specialized flash attention (H800 sm_90)。

在 gluon_attn.py 单头原型基础上扩展:
- grid 从 (cdiv(S,BLOCK_M),) 扩成 (cdiv(S,BLOCK_M), B*H_Q),第二维索引 (batch, q_head)。
- descriptor 全部用 2D:把 [B,H,S,D] 摊平,head/batch 偏移加到 coord 行坐标上。
  * Q/O : [B*H_Q*S, D],行偏移 qh*S
  * V   : [B*H_KV*S, D],行偏移 kvh*S
  * K^T : [B*H_KV*D, S],行偏移 kvh*D
- GQA:kvh = b*H_KV + (h_in // ratio),ratio = H_Q // H_KV。

保留 gluon_attn.py 不动。
"""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared,
    warpgroup_mma, warpgroup_mma_wait,
)

LOG2E = 1.4426950408889634


@gluon.jit
def _kv_producer(k_desc, v_desc, k_empty, k_ready, k_bufs, v_bufs,
                 n_kv_blocks, k_row0, v_row0, BLOCK_N: gl.constexpr):
    """Producer: TMA 搬 K^T([D,BLOCK_N]) 与 V([BLOCK_N,D]) 到 smem 多 buffer。
    k_row0 / v_row0 为当前 head 在摊平张量里的行基址。"""
    nbuf: gl.constexpr = k_bufs.type.shape[0]
    for i in range(n_kv_blocks):
        idx = i % nbuf
        phase = i // nbuf & 1
        e_bar = k_empty.index(idx)
        r_bar = k_ready.index(idx)
        mbarrier.wait(e_bar, phase ^ 1)
        koff = i * BLOCK_N
        mbarrier.expect(r_bar, k_desc.block_type.nbytes + v_desc.block_type.nbytes)
        # K^T 张量 [B*H_KV*D, S]:列为 kv 序列维,行基址 k_row0
        tma.async_copy_global_to_shared(k_desc, [k_row0, koff], r_bar, k_bufs.index(idx))
        # V 张量 [B*H_KV*S, D]:行 = v_row0 + koff
        tma.async_copy_global_to_shared(v_desc, [v_row0 + koff, 0], r_bar, v_bufs.index(idx))


@gluon.jit
def _attn_consumer(q_desc, o_desc, k_empty, k_ready, k_bufs, v_bufs,
                   n_kv_blocks, off_m, q_row0, o_row0, qk_scale,
                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   D: gl.constexpr, CAUSAL: gl.constexpr, num_warps: gl.constexpr):
    """Consumer (default warpgroup): QK^T -> online softmax(exp2) -> P@V。"""
    nbuf: gl.constexpr = k_bufs.type.shape[0]
    m_i_shape: gl.constexpr = 16
    k_dim: gl.constexpr = 16
    wpc: gl.constexpr = [num_warps, 1]

    qk_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=wpc, instr_shape=[m_i_shape, BLOCK_N, k_dim])
    pv_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=wpc, instr_shape=[m_i_shape, D, k_dim])
    q_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_layout, k_width=2)
    p_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_layout, k_width=2)
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    row_pv: gl.constexpr = gl.SliceLayout(1, pv_layout)

    q_smem = gl.allocate_shared_memory(q_desc.dtype, q_desc.block_type.shape, q_desc.layout)
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
        mbarrier.wait(r_bar, phase)

        qk_acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=qk_layout)
        qk = warpgroup_mma(q, k_bufs.index(idx), qk_acc, is_async=True, use_acc=False)
        qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))
        qk = qk * qk_scale

        if CAUSAL:
            offs_n = i * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, qk_layout))
            qk = gl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))

        m_curr = gl.max(qk, axis=1)
        m_new = gl.maximum(m_i, m_curr)
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        acc = acc * gl.convert_layout(alpha, row_pv)[:, None]

        p_op = gl.convert_layout(p.to(gl.bfloat16), p_reg_layout)
        acc = warpgroup_mma(p_op, v_bufs.index(idx), acc, is_async=True, use_acc=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        mbarrier.arrive(e_bar, count=1)
        m_i = m_new

    acc = acc / gl.convert_layout(l_i, row_pv)[:, None]
    o_smem = gl.allocate_shared_memory(o_desc.dtype, o_desc.block_type.shape, o_desc.layout)
    o_smem.store(acc.to(o_desc.dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(o_desc, [o_row0 + off_m, 0], o_smem)
    tma.store_wait(0)


# ======================= warp-specialized kernel =======================
@gluon.jit
def attn_ws_kernel(q_desc, k_desc, v_desc, o_desc, seqlen,
                   qk_scale, H_Q: gl.constexpr, H_KV: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   D: gl.constexpr, CAUSAL: gl.constexpr,
                   NBUF: gl.constexpr, num_warps: gl.constexpr):
    pid_m = gl.program_id(0)
    pid_bh = gl.program_id(1)          # 0 .. B*H_Q-1
    off_m = pid_m * BLOCK_M

    b = pid_bh // H_Q
    qh = pid_bh % H_Q
    ratio: gl.constexpr = H_Q // H_KV
    kvh = qh // ratio

    # 摊平张量的行基址
    q_row0 = (b * H_Q + qh) * seqlen        # Q/O: [B*H_Q*S, D]
    v_row0 = (b * H_KV + kvh) * seqlen      # V  : [B*H_KV*S, D]
    k_row0 = (b * H_KV + kvh) * D           # K^T: [B*H_KV*D, S]

    if CAUSAL:
        n_kv_blocks = (off_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    else:
        n_kv_blocks = (seqlen + BLOCK_N - 1) // BLOCK_N

    k_bufs = gl.allocate_shared_memory(k_desc.dtype, [NBUF] + k_desc.block_type.shape, k_desc.layout)
    v_bufs = gl.allocate_shared_memory(v_desc.dtype, [NBUF] + v_desc.block_type.shape, v_desc.layout)
    k_empty = gl.allocate_shared_memory(gl.int64, [NBUF, 1], mbarrier.MBarrierLayout())
    k_ready = gl.allocate_shared_memory(gl.int64, [NBUF, 1], mbarrier.MBarrierLayout())
    for j in gl.static_range(NBUF):
        mbarrier.init(k_empty.index(j), count=1)
        mbarrier.init(k_ready.index(j), count=1)

    gl.warp_specialize(
        (q_desc, o_desc, k_empty, k_ready, k_bufs, v_bufs, n_kv_blocks, off_m,
         q_row0, q_row0, qk_scale, BLOCK_M, BLOCK_N, D, CAUSAL, num_warps),
        _attn_consumer,
        (k_desc, v_desc, k_empty, k_ready, k_bufs, v_bufs, n_kv_blocks,
         k_row0, v_row0, BLOCK_N),
        [_kv_producer],
        [1], [24])


# ============================ host launcher ============================
def _make_descs(qf, ktf, vf, of, B, H_Q, H_KV, S, BLOCK_M, BLOCK_N, D):
    """qf/of: [B*H_Q*S, D]; vf: [B*H_KV*S, D]; ktf: [B*H_KV*D, S] (K 已转置摊平)。"""
    q_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, D], gl.bfloat16)
    k_layout = gl.NVMMASharedLayout.get_default_for([D, BLOCK_N], gl.bfloat16)
    v_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, D], gl.bfloat16)
    o_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, D], gl.bfloat16)
    q_desc = TensorDescriptor.from_tensor(qf, [BLOCK_M, D], q_layout)
    k_desc = TensorDescriptor.from_tensor(ktf, [D, BLOCK_N], k_layout)
    v_desc = TensorDescriptor.from_tensor(vf, [BLOCK_N, D], v_layout)
    o_desc = TensorDescriptor.from_tensor(of, [BLOCK_M, D], o_layout)
    return q_desc, k_desc, v_desc, o_desc


def _prepare(q, k, v, BLOCK_M=128, BLOCK_N=128, D=128):
    """host 侧预处理:permute/转置/摊平 + 建 descriptor。返回 launch 所需上下文。"""
    B, S, H_Q, _ = q.shape
    H_KV = k.shape[2]
    qc = q.permute(0, 2, 1, 3).contiguous()          # [B, H_Q, S, D]
    kc = k.permute(0, 2, 1, 3).contiguous()          # [B, H_KV, S, D]
    vc = v.permute(0, 2, 1, 3).contiguous()
    qf = qc.reshape(B * H_Q * S, D)
    vf = vc.reshape(B * H_KV * S, D)
    ktf = kc.transpose(-2, -1).contiguous().reshape(B * H_KV * D, S)
    of = torch.empty_like(qf)
    q_desc, k_desc, v_desc, o_desc = _make_descs(qf, ktf, vf, of, B, H_Q, H_KV, S,
                                                 BLOCK_M, BLOCK_N, D)
    return dict(B=B, S=S, H_Q=H_Q, H_KV=H_KV, D=D, of=of,
                descs=(q_desc, k_desc, v_desc, o_desc))


def _run_kernel(ctx, causal=True, BLOCK_M=128, BLOCK_N=128, NBUF=3, num_warps=4):
    """只 launch kernel,不含 host 预处理。输出保持摊平 of。"""
    B, S, H_Q, H_KV, D = ctx["B"], ctx["S"], ctx["H_Q"], ctx["H_KV"], ctx["D"]
    q_desc, k_desc, v_desc, o_desc = ctx["descs"]
    qk_scale = (1.0 / (D ** 0.5)) * LOG2E
    grid = (triton.cdiv(S, BLOCK_M), B * H_Q)
    attn_ws_kernel[grid](q_desc, k_desc, v_desc, o_desc, S, qk_scale,
                         H_Q, H_KV, BLOCK_M, BLOCK_N, D, causal, NBUF,
                         num_warps=num_warps, maxnreg=128)


def attn_ws_mha(q, k, v, causal=True, BLOCK_M=128, BLOCK_N=128, NBUF=3, num_warps=4):
    """端到端:输入/输出均为 [B, S, H, D](与生产 kernel 一致)。"""
    ctx = _prepare(q, k, v, BLOCK_M, BLOCK_N, q.shape[-1])
    _run_kernel(ctx, causal, BLOCK_M, BLOCK_N, NBUF, num_warps)
    B, S, H_Q, D = ctx["B"], ctx["S"], ctx["H_Q"], ctx["D"]
    return ctx["of"].reshape(B, H_Q, S, D).permute(0, 2, 1, 3).contiguous()


# ============================ reference & test ============================
def ref_attn(q, k, v, causal=True):
    """q:[B,S,H_Q,D]  k/v:[B,S,H_KV,D]  GQA 广播。返回 [B,S,H_Q,D]。"""
    B, S, H_Q, D = q.shape
    H_KV = k.shape[2]
    ratio = H_Q // H_KV
    qf = q.permute(0, 2, 1, 3).float()               # [B,H_Q,S,D]
    kf = k.permute(0, 2, 1, 3).float()               # [B,H_KV,S,D]
    vf = v.permute(0, 2, 1, 3).float()
    # 广播 kv 头
    kf = kf.repeat_interleave(ratio, dim=1)          # [B,H_Q,S,D]
    vf = vf.repeat_interleave(ratio, dim=1)
    scores = (qf @ kf.transpose(-2, -1)) / (D ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = (p @ vf)                                     # [B,H_Q,S,D]
    return o.permute(0, 2, 1, 3).to(q.dtype)         # [B,S,H_Q,D]


def _make_qkv(B, S, H_Q, H_KV, D, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, S, H_Q, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H_KV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H_KV, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _bench(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def _prod_call(q, k, v):
    """生产 kernel:flag_gems mha_fwd,输入 [B,S,H,D]。"""
    from flag_gems.ops import flash_api
    out = torch.empty_like(q)
    scale = 1.0 / (q.shape[-1] ** 0.5)
    flash_api.mha_fwd(q, k, v, out, None, 0.0, scale, True, -1, -1, 0.0, False)
    return out


if __name__ == "__main__":
    D = 128
    print("=== gluon WS 多头/GQA flash attention (H800, bf16, d128, causal) ===\n")

    # 步骤1:标准 MHA (H_Q=H_KV=28),验证占用率假设
    print("--- 步骤1: 标准 MHA (H=28, B=1), 占用率验证 ---")
    for S in [4096, 8192]:
        q, k, v = _make_qkv(1, S, 28, 28, D)
        out = attn_ws_mha(q, k, v, True)
        err = (out.float() - ref_attn(q, k, v, True).float()).abs().max().item()
        ctx = _prepare(q, k, v)
        t_k = _bench(lambda: _run_kernel(ctx, True))       # 仅 kernel
        t_e = _bench(lambda: attn_ws_mha(q, k, v, True))   # 端到端(含转置)
        naive = 215.72 if S == 4096 else 414.99
        print(f"  S={S:5d}: err={err:.5f} {'OK' if err<0.05 else 'FAIL'}  "
              f"kernel={t_k:7.1f}us  端到端={t_e:7.1f}us  "
              f"(单头×28朴素预期={naive*28:.0f}us)")

    # 步骤2/3:GQA (H_Q=28, H_KV=4),Qwen2.5-7B 真实形状,同口径对比生产 kernel
    print("\n--- 步骤2/3: GQA (H_Q=28, H_KV=4, B=1), 同口径对比生产 kernel ---")
    print(f"  {'seqlen':>6} | {'生产kernel':>10} | {'gluon-WS-kernel':>15} | "
          f"{'gluon-WS-端到端':>15} | {'比值(kernel)':>11}")
    for S in [4096, 8192]:
        q, k, v = _make_qkv(1, S, 28, 4, D)
        out = attn_ws_mha(q, k, v, True)
        ref = ref_attn(q, k, v, True)
        prod = _prod_call(q, k, v)
        err_g = (out.float() - ref.float()).abs().max().item()
        err_p = (prod.float() - ref.float()).abs().max().item()
        ctx = _prepare(q, k, v)
        t_prod = _bench(lambda: _prod_call(q, k, v))
        t_k = _bench(lambda: _run_kernel(ctx, True))
        t_e = _bench(lambda: attn_ws_mha(q, k, v, True))
        print(f"  {S:6d} | {t_prod:9.1f}us | {t_k:14.1f}us | {t_e:14.1f}us | "
              f"{t_k/t_prod:10.2f}x")
        print(f"         正确性: gluon err={err_g:.5f} {'OK' if err_g<0.05 else 'FAIL'}, "
              f"生产 err={err_p:.5f} {'OK' if err_p<0.05 else 'FAIL'}")

