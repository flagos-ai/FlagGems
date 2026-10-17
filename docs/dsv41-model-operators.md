# Framework-independent V4.1 model operators

These operators accept tensors and documented numeric layouts. They do not
import SGLang. Model-specific computation belongs here even when it uses
TileLang; framework request allocators and candidate layouts remain in
[FlagGems-sglang #161](https://github.com/flagos-ai/FlagGems-sglang/pull/161).
Block-FP8 GEMM is maintained separately in
[FlagGems #6631](https://github.com/flagos-ai/FlagGems/pull/6631).

## Packed FP4 decoding and explicit-slot index scoring

`flag_gems.fused.dsv41.fp4_indexer.fp4_index_logits_decode(q, weights, slots, lens, table, page_size)`
accepts BF16 q `[B,H,128]`, weights `[B,H]` (converted to BF16), explicit
physical slots `[B,L]` int32/int64, visible lengths `[B]`, and a byte table
`[num_pages, page_size*68]`. Slots address this explicit storage only; there
is no request-to-token table, compression ratio, candidate ID or allocator.
Inputs share one CUDA device; H100/SM90 is the validated target.

Each page stores all `page_size*64` payload bytes followed by all
`page_size*4` scale bytes. In each payload byte the low nibble is the even
E2M1 value and the high nibble is the odd value. Magnitudes are
0/.5/1/1.5/2/3/4/6, with a sign bit; negative zero is canonicalized to
positive zero. Each UE8M0 scale byte covers 32 values and denotes
`2**(byte-127)`. The supported input contract uses finite scale bytes
0–254; byte 255, invalid physical slots and NaN inputs are outside it.
Callers own table capacity and valid per-replay slots/lengths.

Decoded keys become BF16 before the dot. The two even/odd BF16 dots use
FP32 accumulation, then preserve BF16 dot rounding, ReLU, BF16 weighted
product rounding, and BF16 head-sum rounding. Output is FP32 `[B,L]`,
with `-inf` beyond visible lengths. Output storage may pad columns for
alignment. Empty width returns `[B,0]` without a launch.

The Triton `_e2m1_decode` and `fp4_weighted_head_score` helpers contain the
shared numeric computation. SGLang-specific fused kernels inline these
helpers around their own addressing and candidate writes; this introduces
no intermediate tensors or extra GPU launches. Their internal contract is
versioned by `FP4_INDEX_SCORE_ABI=1`.

## mHC statistics, combine and exact Sinkhorn

`flag_gems.fused.dsv41.mhc` owns `hc_mix_stats`, `hc_combine`,
`hc_split_sinkhorn` and `hc_mix_stats_sinkhorn_exact`.

Statistics accept `[M,K]` activations and FP32 `[MIX,K]` weights. K20480
uses deterministic 80-way slicing with K64/TF32x3 partials; changing the
batch row count does not change reduction order. The combine preserves
the predecessor BF16 rounding. Exact fusion combines the final reduction
with the original TileLang Sinkhorn, retaining all original rounding
boundaries. It supports HC4 at M36/40/42/48/50/60/80/84/96; other rows use
separate operators. There is no approximate Sinkhorn replacement.

TileLang 0.1.12 is required only when an exact Sinkhorn entry is called;
its import/compilation is lazy. `MHC_EXACT_ABI=1` describes this interface.
The old unrelated block-FP8 conversion and approximate fusion files are
removed from this PR. FP4 and mHC changes are separate review commits.

Run from a CUDA checkout with `PYTHONPATH=src`:

```bash
python -m pytest -q tests/test_dsv41_fp4_core.py tests/test_dsv41_mhc.py
python benchmark/dsv41_mhc_graph.py --out mhc-graph.json --rounds 2 --rep-ms 100
```

Tests use independent Torch references, byte-exact regression and changed
CUDA Graph inputs. The pre-split mHC microbenchmark measured 24 logical
serving rows, geometric mean 1.084x, range 1.000–1.285x, retaining unchanged
rows and measurement noise. The immutable numerical source is
[the measured pre-split port](https://github.com/CherryLemon/FlagGems-sglang/commit/b188ef6b9de41138331299651aa23f3e11fb8187).
Framework FP4 fusion speedups are documented in its owning repository,
not attributed to the standalone explicit-slot API here.
