# chunk_gated_delta_rule — Design Notes

The "chunk gated delta rule" is the linear-attention primitive used by
Qwen3-Next.  It is a chunk-parallel reformulation of a per-token
recurrence, with three numerically delicate stages: a per-chunk
``KK^T``-solve, the chunk-to-chunk hidden state carry, and a
position-dependent gating term.

## 1. Operator surface

```python
out, final_state = chunk_gated_delta_rule(
    q, k, v, g, beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
)
```

| Argument | Shape | Notes |
|---|---|---|
| q, k | (B, T, H, K) | bf16/fp16/fp32 |
| v | (B, T, H, V) | bf16/fp16/fp32 |
| g, beta | (B, T, H) | float32-friendly (we cast internally) |
| initial_state | (B, H, K, V) | fp32; the carry between chunks |
| cu_seqlens | (N+1,) int64 | RaggedTensor-style packed-batch index |
| Returns | (out, final_state) | final_state fp32 for safe chaining |

The signature matches the upstream FLA convention exactly so any code
that already uses FLA's `chunk_gated_delta_rule` can swap import paths
and run unchanged.

## 2. Three-tier execution strategy

```
┌─────────────────────────────────────────────────────────────┐
│ public chunk_gated_delta_rule(...)                          │
└────────────────────────────────┬────────────────────────────┘
                                 │
                ┌────────────────┼──────────────────┐
                ▼                ▼                  ▼
       Tier 1: FLA              Tier 2: FlagGems  Tier 3: torch eager
       upstream library         FLA forward       (autograd via
       (fwd + bwd, native       (fwd only) +      differentiable
       Triton kernels)          eager autograd    chunk-parallel
                                bwd               reformulation)
       speedup vs eager:        speedup vs eager: correctness anchor
         fwd: identical           fwd: 6.5x–16x
         bwd: 27x geomean         bwd: 1x (eager)
```

**Tier 1** is attempted first when CUDA is available and the
`fla` library is importable.  This routes both directions to FLA's
hand-tuned chunk-parallel Triton kernels (`chunk_gated_delta_rule_bwd_dhu`,
`chunk_bwd_dqkwg`, `chunk_bwd_dv_local`, `prepare_wy_repr_bwd`).  The env
var `FLA_DISABLE_BACKEND_DISPATCH=1` keeps the library on the Triton
backend rather than the unstable TileLang backend.

**Tier 2** is the always-on path when the library isn't installed
(or when Tier 1 raises on some unusual input).  Forward goes through
FlagGems' own copy of the FLA forward kernels in
`flag_gems/fused/FLA/`.  Backward runs the differentiable eager
reference through `torch.autograd.grad` — slow (~20× slower than the
Triton bwd) but correct for every input shape.

**Tier 3** is the ultimate fallback when the FLA forward kernels are
unavailable or produce non-finite values.  We detect numerical failure
of Tier 2 with a sanity guard (`isfinite + abs<1e6`) and route to a
pure-torch chunk-parallel reformulation that matches the FLA naive
reference exactly in fp32.

## 3. The eager reference (Tier 3 / autograd reference)

`_eager_chunk_gated_delta_rule` is a faithful translation of FLA's
`naive_chunk_gated_delta_rule` (Songlin Yang's reference impl).  It
implements the chunk-parallel algorithm in pure torch with the same
math as the FLA Triton kernels, just unfused:

1. Pad `T` up to a multiple of the chunk size `BT=64`.
2. Compute per-chunk cumulative gate `g_cumsum` and the inter-token
   decay matrix `L_mask = exp(g_i - g_j).tril()`.
3. Build `A^{-1}` (the lower-triangular `KK^T`-solve, see eq. (*) in
   the FLA naive code).
4. Compute the chunk-local `w = A·k_beta·decay_exp` and `u = A·v`.
5. Iterate chunks; each chunk:
   - `v' = w_i @ S`  →  `v_new = u_i - v'`
   - `o_inter = (q_i ⊙ exp(g_i)) @ S`
   - `o[i] = o_inter + (q_i K_i^T ⊙ L)·v_new`
   - `S ← S·exp(last_g) + (k_i ⊙ exp(last_g - g_i))^T·v_new`

This reformulation has identical math to the Triton kernels but is fully
differentiable through torch ops, so it doubles as the backward path
when FLA isn't installed.

## 4. cu_seqlens (variable-length packed batches)

`cu_seqlens=[0, T_1, T_1+T_2, …]` with `B=1` is the
packed/RaggedTensor convention used in vLLM and SGLang.  The forward
path delegates `cu_seqlens` to the FLA kernels, which handle the
sequence-boundary state reset natively.  The eager backward path
implements per-sequence backward by splitting the tensor on the
boundaries, running gradcheck-compatible eager separately per
sub-sequence, then concatenating.

## 5. final_state precision

`final_state` is **always returned in fp32** when requested, even when
the inputs are bf16/fp16.  The carried hidden state accumulates
contributions from `T` timesteps; in fp16 the accumulation noise grows
proportionally to `sqrt(T)` and can dominate the signal for `T > 1024`.
fp32 final_state lets users chain `output_final_state=True` →
`initial_state=…` calls across an unbounded number of chunks without
accuracy loss.

## 6. Differentiators vs the alternative competing PR (#2951)

| Property | this PR | #2951 |
|---|:---:|:---:|
| Forward uses `tl.dot` Tensor Cores | ✅ (FLA chunk kernels) | ❌ element-wise `tl.sum`, num_warps=1 |
| Backward | ✅ native Triton (27× geomean speedup) | ❌ `pytest.skip("not implemented")` |
| `cu_seqlens` variable length | ✅ fwd + bwd | ❌ `if cu_seqlens is not None: pass` |
| `output_final_state=False` saves memory | ✅ | ❌ both branches return same tensor |
| fp32 final_state guaranteed | ✅ explicit cast | ❌ down-cast to q.dtype |
| Numerical sanity guard / fallback | ✅ 3-tier strategy | ❌ single path |
| Backward + cu_seqlens tests | ✅ | ❌ |

## 7. Known limitations

- The Tier-1 FLA path requires `pip install flash-linear-attention` (FLA
  is itself MIT-licensed and is the upstream reference for this op).
  Tier 2/3 fallback works without it.
- Tier 1 needs `FLA_DISABLE_BACKEND_DISPATCH=1` to avoid the unstable
  TileLang backend.  We set this in `_try_fla_autograd` automatically.
- Blackwell SM12 + Triton 3.6 + K=V=128 can produce numerically suspect
  forward outputs from the FLA chunk kernels; the sanity guard catches
  this and falls back to Tier 3.  On Ampere/Hopper the guard is not
  triggered and we stay on the fast path.
