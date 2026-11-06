# Framework-independent V4.1 model operators

These operators accept tensors and documented numeric layouts. They do not
import SGLang. Model-specific computation belongs here even when it uses
TileLang; framework request allocators and candidate layouts remain in
[FlagGems-sglang #194](https://github.com/flagos-ai/FlagGems-sglang/pull/194).
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

## Guarded FP4-storage / FP8-compute helpers

`flag_gems.fused.dsv41.fp4_fp8` adds the independent query preparation,
E2M1-to-E4M3 conversion, block-scaled FP8 dot and predecessor rounding
helpers under `FP4_FP8_SCORE_ABI=2`. Storage remains packed FP4 with UE8M0
scales. The caller owns addressing, grouping, workspace, runtime eligibility
and the BF16 fallback; these helpers do not import a serving framework.

The exact fast path is bounded: normalized query blocks must satisfy the
query-preparation multiples-of-16/exponent-span checks, and key scale
encodings must satisfy the documented dot helper contract. Recompute these
checks on every graph replay. Unsupported values retain the existing BF16
calculation; a successful benchmark on eligible inputs does not imply that
every arbitrary BF16 query can use FP8.

The Gluon `e2m1x4_to_fp8` intrinsic uses a four-element register pack. The
caller must arrange each group so its input elements refer to the same
16-bit packed word; the intrinsic expands that word into four E4M3 bytes.
The validated Gluon layout supplies this invariant. A different DSL/layout
must establish it or use a layout-independent packed-word conversion.

`fp4_fp8_tle.decode_fp4_e2m1_words_to_fp8x4` provides that independent
conversion: each uint32 input lane's low 16 bits hold four consecutive E2M1
nibbles (widen uint16 loads before calling); one uint32 result holds four
E4M3 bytes in the same order. Both zero signs
become positive zero. This integer helper has no TLE import or register-pack
assumption. Its test exhausts all 65,536 words for uint16/int32 input and
checks changed-input graph replay. Framework-specific TLE shared-memory
layouts and fused request addressing remain in FlagGems-sglang.

H100 qualification covers Triton 3.6.0 and 3.7.1. The exhaustive decode,
query-guard and scale-bit tests are in `tests/test_dsv41_fp4_fp8_math.py`.
Framework-fused scoring and its consumer-chain results remain documented
in FlagGems-sglang.

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

On H100 the contiguous BF16 `[50, 20480]` and `[60, 20480]` statistics
paths with `[24, 20480]` FP32 weights use a capped register launch. Other
inputs retain their existing launch settings.


## Common prefill and MoE math

The new standalone modules contain only explicit tensors and numeric
contracts, with no SGLang model objects, request tables or scheduler state:

* `fp8_prefill_score`: E4M3 Q/K, BF16 head weights, device visibility lengths.
  Dot, weighted-ReLU and head reduction retain their BF16 round points. The
  eight-query reuse path uses lossless FP16 operand casts on the qualified
  compiler; FP8 storage alone is not a native FP8-MMA claim.
* `mhc_post_combine`: emits both the materialized BF16 post residual and BF16
  combined input, preserving FP32 FMA order. Pre coefficients must already
  be known at the post boundary. The public selected path uses direct loads.
* `mhc_prefill_stats`: per-call K-major high/low TF32 weight preparation, with
  the original 80-slice reduction order and no cached tensor/workspace.
* `fp4_prefill`: converts BF16 queries and gathers explicit packed FP4 slots
  into FP8, and computes dense FP8 index scores for general head counts.
* `masked_row_quant`: explicit expert/capacity rows and device valid counts;
  optional row splits select the compact scheduling variant without changing
  the physical output layout.
* `masked_swiglu`: clipped gate/up activation on valid expert rows, with a
  caller-supplied Triton activation callback and caller-owned output.
* `swiglu_row_quant`: clipped gate/up activation, BF16 materialization and
  row quantization. The caller supplies original calc/finalize/conversion JIT
  callbacks. Invalid route rows and output padding remain untouched.

The callback interfaces deliberately avoid a dependency on Humming; its source
qualification and SGLang route layout belong to FlagGems-sglang. Host wrappers
reject unsupported layouts before launch. The caller owns output buffers and
must provide non-overlapping storage on the same device.
