"""vfwmacc GQA attention kernel for SpacemiT K3 (arch 0xA064, vlen=1024).

Fused M=1 decode kernel (attn_m1_fused_raw): QKT dots + 2-phase softmax +
AV in one launch. Per-head grid (Hq,) with GQA via hk = hq // Nrep.
D=128 handled by 2x 64-chunk (Qwen3 head_dim). f16 input -> f32 acc
(vfwmacc widening) -> fptrunc -> f16 output. Default-path eDSL codegen
(vexp/vmacc/vreduce) — no math intrinsics (LLVM 23 removed
llvm.intr.exp/maxnum).
Prefill (Mq>1) lives in flash_attention.py (_attn_fwd, smt.dot flash).
Validated on Qwen3-0.6B (K3): decode Mk=4096 2.99ms vs native 96.4ms.
"""
import triton
import triton.language as tl
import triton.language.extra.smt_rvisa as tle
from triton.language.extra.smt_rvisa import call as _sr_call
from triton.language.extra.smt_rvisa import f16, f32, index, mem


# ── Fused M=1: QKT + softmax + AV in one kernel (decode-only) ───────────────
# Default-path spine_raw (vexp/vmacc/vreduce). grid=(Hq,), host passes hq/hk
# as index scalars (multi-core via JIT wrapper, _to_handle casts i32→index).
# Decode Mq=1: no causal mask (query at pos Mk-1 attends to all keys 0..Mk-1).
# 2-phase softmax: dots→scratch (Hq,Mk) f32 → vreduce_max → vexp → normalize → AV.
# D=128 = 2× 64-chunk (Qwen3 head_dim). f16 input → f32 acc → f16 output.
@tle.raw_kernel
def attn_m1_fused_raw(
    Q_ptr: mem(f16),
    K_ptr: mem(f16),
    V_ptr: mem(f16),
    Scratch_ptr: mem(f32),
    Scale_ptr: mem(f32),
    O_ptr: mem(f16, out=True),
    Mk: index,
    D: index,
    hq: index,
    hk: index,
):
    tle.vconfig(-1, 1)  # VL=64 (f16/f32 element count, lmul=1)

    scale = tle.sload(Scale_ptr, 0, dtype=f32)

    MkD = Mk * D

    # Load Q[hq, 0:64] and Q[hq, 64:128] as f32 (hoisted)
    qv0_h = tle.vload(Q_ptr, hq * D, dtype=f16)
    qv0 = tle.cast(qv0_h, f32)
    qv1_h = tle.vload(Q_ptr, hq * D + 64, dtype=f16)
    qv1 = tle.cast(qv1_h, f32)

    # Phase 1: QKT dots → scratch (default-path: cast + mul + vreduce_sum)
    for k in tle.range(0, Mk, 1):
        kv0_h = tle.vload(K_ptr, hk * MkD + k * D, dtype=f16)
        kv0 = tle.cast(kv0_h, f32)
        prod0 = qv0 * kv0
        dot = tle.vreduce_sum(prod0)
        kv1_h = tle.vload(K_ptr, hk * MkD + k * D + 64, dtype=f16)
        kv1 = tle.cast(kv1_h, f32)
        prod1 = qv1 * kv1
        dot = dot + tle.vreduce_sum(prod1)
        dot = dot * scale
        tle.sstore(Scratch_ptr, hq * Mk + k, dot)

    # Phase 2: global max (default-path, non-scalable f32)
    nvl2 = tle.vconfig(-1, 2)  # VL=128 for faster reduce on small Mk
    Kfloor = (Mk // nvl2) * nvl2
    max_vec = tle.vzero(f32) - 1.0e38
    for k in tle.range(0, Kfloor, nvl2):
        v = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32)
        max_vec = tle.vmax(max_vec, v)
    for k in tle.range(Kfloor, Mk, nvl2):
        _ = tle.vconfig(Mk - k, 2)
        vt = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32, fill=-1.0e38)
        max_vec = tle.vmax(max_vec, vt)
    _ = tle.vconfig(-1, 2)
    max_s = tle.vreduce_max(max_vec)

    # Phase 3a: exp → scratch, accumulate sum (default-path)
    sum_vec = tle.vzero(f32)
    for k in tle.range(0, Kfloor, nvl2):
        v = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32)
        ev = tle.vexp(v - max_s)
        tle.vstore(Scratch_ptr, hq * Mk + k, ev)
        sum_vec = sum_vec + ev
    for k in tle.range(Kfloor, Mk, nvl2):
        _ = tle.vconfig(Mk - k, 2)
        vt = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32, fill=-1.0e38)
        evt = tle.vexp(vt - max_s)
        tle.vstore(Scratch_ptr, hq * Mk + k, evt)
        sum_vec = sum_vec + evt
    _ = tle.vconfig(-1, 2)
    sum_exp = tle.vreduce_sum(sum_vec)
    inv = 1.0 / sum_exp

    # Phase 3b: normalize exp in scratch by inv (default-path)
    for k in tle.range(0, Kfloor, nvl2):
        ev = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32)
        tle.vstore(Scratch_ptr, hq * Mk + k, ev * inv)
    for k in tle.range(Kfloor, Mk, nvl2):
        _ = tle.vconfig(Mk - k, 2)
        ev = tle.vload(Scratch_ptr, hq * Mk + k, dtype=f32, fill=0.0)
        tle.vstore(Scratch_ptr, hq * Mk + k, ev * inv)
    _ = tle.vconfig(-1, 1)

    # Phase 4: AV (default-path: vv_f * ek scalar broadcast, accumulate)
    acc0 = tle.vzero(f32)
    acc1 = tle.vzero(f32)
    for k in tle.range(0, Mk, 1):
        ek = tle.sload(Scratch_ptr, hq * Mk + k, dtype=f32)
        vv0_h = tle.vload(V_ptr, hk * MkD + k * D, dtype=f16)
        vv0 = tle.cast(vv0_h, f32)
        acc0 = acc0 + vv0 * ek
        vv1_h = tle.vload(V_ptr, hk * MkD + k * D + 64, dtype=f16)
        vv1 = tle.cast(vv1_h, f32)
        acc1 = acc1 + vv1 * ek

    # Phase 5: trunc + store (default-path: cast f32→f16, vstore)
    out0 = tle.cast(acc0, f16)
    tle.vstore(O_ptr, hq * D, out0)
    out1 = tle.cast(acc1, f16)
    tle.vstore(O_ptr, hq * D + 64, out1)


@triton.jit(do_not_specialize=["Mk", "D"])
def _attn_m1_fused_host(
    Q, K, V, Scratch, Scale, Out, Mk, D, Nrep: tl.constexpr, Hq: tl.constexpr
):
    hq = tl.program_id(0)
    hk = hq // Nrep
    _sr_call(
        attn_m1_fused_raw,
        outputs=[],
        inputs=[Q, K, V, Scratch, Scale, Out, Mk, D, hq, hk],
    )
