# MTHREADS Conv1D Padding Benchmark Optimization Investigation

## Scope

- **Performance target**: `benchmark/test_conv1d.py -m conv1d_padding`
- **Underlying operator**: `flag_gems.conv1d` (canonical Conv1D via Conv2D-via-unsqueeze)
- **Prototype**: MTHREADS FP16 true-1D implicit-GEMM Triton kernel
- **Production result**: Prototype not landed (docs-only investigation)

## Summary

`conv1d_padding` is a benchmark/pytest scope, not a separate FlagGems
operator. The underlying operator is `flag_gems.conv1d`, whose canonical
implementation reuses Conv2D via unsqueeze/squeeze.

A dedicated FP16 true-1D implicit-GEMM Triton prototype was evaluated
against the canonical Conv2D-via-unsqueeze path on MTT S5000 to determine
whether the Conv1D padding benchmark workload distribution justified a
production MTHREADS Conv1D override.

The prototype showed a strong local win on the representative K7g2 dispatch-
hit case, but no sufficiently broad, regression-free production dispatch
region was established across the evaluated padding benchmark shapes.

The dedicated prototype was therefore not landed. The canonical Conv1D
path remains the production default.

## Baseline Architecture

Input `[N, C_in, L_in]` \
unsqueeze(-1) → `[N, C_in, L_in, 1]` \
→ Conv2D `[N, C_out, L_out, 1]` (kernel `[K, 1]`) \
→ squeeze(-1) → Output `[N, C_out, L_out]`

The canonical path is `flag_gems.ops.conv1d`
(src/flag_gems/ops/conv1d.py), which always delegates to Conv2D via
unsqueeze/squeeze. `unsqueeze`/`squeeze` are view operations and are not
themselves the dominant performance cost; the cost is in the underlying
Conv2D kernel execution.

The canonical path uses integer, `"valid"`, or `"same"` padding through
the Conv2D backend, with dilation and groups forwarded transparently.

## Candidate Architecture

A hand-written MTHREADS Triton prototype evaluated during the investigation
(historical reference: PR #6027). Key features:

- **True 1D implicit-GEMM**: no per-tap 4D address arithmetic; direct 1D
  addressing via `tl.arange` over the L axis and C axis.
- **K-tap loop**: `tl.static_range(0, kernel_size)` over the receptive
  field.
- **`tl.dot`**: implicit-GEMM accumulation into `[BLOCK_L, BLOCK_CO]`
- **Hard-coded tile config**: `(BLOCK_L, BLOCK_CO, BLOCK_CI) = (128, 32, 16)`
  with `num_warps=8, num_stages=1`. No `@triton.autotune`.
- **FP16 only**: FP32 always falls through to the canonical path.
- **Same-padding fallback**: `padding == "same"` always routed to canonical
  because the dedicated kernel only supports symmetric left padding and
  even-K "same" requires asymmetric padding.
- **Structural dispatch heuristic** (candidate, not production):
  `in_c_per_group <= 32 OR (kernel_size >= 7 AND in_l >= 2048)`

This was a prototype only, not production code.

## Methodology

Benchmark target: `benchmark/test_conv1d.py -m conv1d_padding`

Timing method: `triton.testing.do_bench`, the same method used by the
official FlagGems Conv1D benchmark, configured with warmup (200) and
repetition (500) per `Config` settings.

Process isolation: each shape measured in a fresh process with a fresh
Triton compilation cache to prevent cache pollution and attribution bias.

Measurement modes:

- **forced**: dedicated prototype manually forced for investigation;
  not a production routing path.
- **dispatch-hit**: candidate structural dispatch selects the dedicated
  prototype.
- **fallback**: candidate routing selects the canonical implementation.

Hardware: MTT S5000.

> **Ratio definitions**:
> - `canonical_gems_speedup = torch_us / canonical_us`
> - `prototype_gems_speedup = torch_us / dedicated_us`
> - `prototype_vs_canonical = canonical_us / dedicated_us`
>
> `canonical_gems_speedup > 1.0` means the canonical Gems path is faster
> than Torch. `prototype_vs_canonical > 1.0` means the prototype is
> faster than canonical.

## Performance

**IMPORTANT**: The dedicated measurements below are investigation/prototype
results. They do not represent production FlagGems performance because the
dedicated Conv1D kernel was not landed. Canonical Gems Speedup represents
the current production path.

#### FP16 investigation results

| measurement_mode | shape | torch_us | canonical_us | prototype_us | canonical_gems_speedup | prototype_gems_speedup | prototype_vs_canonical |
|---------------|-------|---------:|-------------:|-------------:|----------------------:|----------------------:|----------------------:|
| forced | K3 32×64×512, 64×64×3, g=1, fp16 p=1 | 30.700 | 70.400 | 70.100 | 0.436x | 0.438x | 1.004x |
| forced | K5 64×48×1024, 128×48×5, g=1, s=2, fp16 p=2 | 47.000 | 245.800 | 290.000 | 0.191x | 0.162x | 0.848x |
| dispatch-hit | K7g2 16×24×2048, 96×12×7, g=2, fp16 p=3 | 73.000 | 148.500 | 79.400 | 0.492x | 0.919x | 1.870x |
| fallback | K11 8×8×8192, 16×8×11, g=1, fp16 p=5 (same) | 80.000 | 140.000 | 140.000 | 0.571x | 0.571x | 1.000x |

Definitions:

- **canonical_gems_speedup** = torch_us / canonical_us
- **prototype_gems_speedup** = torch_us / prototype_us
- **prototype_vs_canonical** = canonical_us / prototype_us (>1 = prototype faster)

Mode explanation:

- **forced**: dedicated prototype was manually forced for investigation;
  not production routing
- **dispatch-hit**: candidate structural dispatch actually selects dedicated
  prototype
- **fallback**: candidate routing selects canonical implementation

The table includes forced prototype measurements for K3/K5, an actual
dispatch-hit measurement for K7g2, and a canonical fallback measurement
for K11. K3 is near parity when the prototype is forced, while K5 shows a
material regression (0.848x). Neither K3 nor K5 would hit the candidate
production dispatch rule.

Units: us.

## Interpretation

### K3

Near parity in the forced measurement (1.004x). The prototype does not
gain over the canonical path on this representative case. K3 does not
satisfy `in_c_per_group <= 32` (in_c/g = 64) and does not satisfy
`kernel_size >= 7 AND in_l >= 2048` (kernel=3, in_l=512), so the candidate
dispatch would route it to canonical.

### K5

Material regression in the forced measurement (0.848x, ~15% slower). The
prototype is significantly slower than canonical on this representative
padded case. K5 does not satisfy the candidate dispatch heuristic
(in_c/g = 48 > 32, kernel=5 < 7), so the candidate routing would select
canonical.

### K7g2

On the representative K7g2 dispatch-hit case, the prototype achieved
1.87x speedup over canonical (canonical_us=148.500 → prototype_us=79.400).
This is a clear win in this single benchmark configuration.

However, this is an isolated result in a single benchmark configuration,
not a broad region.

### K11

Same-padding falls back to canonical by design; no prototype advantage was
observed.

## Decision

The prototype demonstrated real workload-specific headroom on the K7g2
case, but the evaluated Conv1D padding benchmark distribution did not
establish a sufficiently broad, regression-free production dispatch region.

Therefore:

- The dedicated Conv1D prototype was not landed.
- The canonical Conv2D-via-unsqueeze Conv1D path remains the production
  default.
- No functional production code was changed.

**Follow-up opportunity**: future optimization, if pursued, should examine
the underlying Conv2D execution path for FP16 performance improvements
that would benefit the broader Conv1D padding workload distribution.

## Validation

- 28/28 relevant `tests/test_conv1d.py` cases passed against the upstream
  canonical implementation during the final investigation run
  (`conv1d`, `conv1d_padding`, `conv1d_dilation`).
- Benchmark methodology:
  `triton.testing.do_bench` with warmup=200, iter=500, in a fresh process
  with fresh cache, consistent with the official FlagGems kernel benchmark.
- Ratio definitions (auto-computed):
  - canonical_gems_speedup = torch_us / canonical_us
  - prototype_gems_speedup = torch_us / prototype_us
  - prototype_vs_canonical = canonical_us / prototype_us (>1 = prototype faster)

## Benchmark Provenance

- **TORCH** (reference): `torch.nn.functional.conv1d`
- **BASE / Canonical Gems** (production): `flag_gems.conv1d` via
  `flag_gems.ops.conv1d` → Conv2D-via-unsqueeze path.
- **CANDIDATE / Prototype** (investigation): Hand-written 1D implicit-GEMM
  Triton kernel (FP16), referenced in PR #6027. Candidate dispatch
  condition: `in_c_per_group <= 32 OR (kernel_size >= 7 AND in_l >= 2048)`.

The prototype was NOT landed. Canonical Gems represents the production path.

## Reproducibility / Scope

These results are specific to:

- MTT S5000 hardware
- The official `conv1d_padding` benchmark distribution (13 shapes)
- 3 of the 13 shapes were evaluated with the dedicated prototype (K3, K5, K7g2 \
  dispatch-hit); 1 additional shape (K11) was measured in fallback mode

The investigation was scoped to the official Conv1D padding test set;
broader shape sweeps were outside scope. Future compiler or runtime changes
may alter the relative performance, and any dedicated kernel would need
re-evaluation in that context.
