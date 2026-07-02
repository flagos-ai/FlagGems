# Hopper FA3 TLE Kernels

This package contains the active Hopper FA3 TLE forward kernels used through
`ops/flash_kernel_v3.py` and `flash_api_v3.py`.  Decode-only prototypes that
are not part of the current dispatch path live in `bak/`.

## Active Dispatch Surface

| File | Role |
| --- | --- |
| `kernels.py` | Public re-export layer for the active FA3 kernels. |
| `planning.py` | Minimal vLLM-style metadata dispatch helper. |
| `utils.py` | Shared masks, ALiBi, paged gather, TLE copy/barrier helpers, and autotune helpers. |
| `fa_hopper_persistent_pingpong.py` | Long/prefill kernel used by metadata prefill dispatch. |
| `fa_hopper_short.py` | Short prefill kernel used by metadata prefill dispatch. |
| `fa_hopper_direct.py` | Direct decode kernel used when cache-KV metadata is present without split-KV. |
| `fa_hopper_decode_flashdecoding.py` | Flash-Decoding split-KV path used when cache-KV metadata has `num_splits > 1`. |

## Metadata Dispatch

The default FA3 path follows vLLM-style metadata semantics instead of routing
by a best-known workload fallback table:

- dense vs paged: `block_table is None` vs a valid block table
- prefill: `max_query_len > 1` and no cache-KV length tensor
- direct decode: cache-KV lengths are present and the effective split count is 1
- multi-token/speculative decode: `max_query_len > 1` and cache-KV lengths are present
- split-KV / Flash-Decoding: cache-KV lengths are present and the effective split count is greater than 1

When cache-KV lengths are present and the caller passes `num_splits=0`, the
launcher can compute vLLM-style scheduler metadata.  By default this metadata
does not force the current TLE split-KV kernel, because that kernel is not yet
equivalent to vLLM's tile scheduler implementation and can regress paged/decode
workloads.  Set `FLAG_GEMS_FA3_TLE_AUTO_SPLITKV=1` to opt into the experimental
metadata-driven split-KV route.

Dynamic per-sequence split counts are generated and logged.  The active TLE
split-KV kernel still uses the maximum split count as its static launch shape,
with per-batch dynamic split counts only used to skip extra split programs.

Debug/override knobs:

```bash
FLAG_GEMS_FA3_TLE_LOG_PLAN=1           # print metadata/kernel information
FLAG_GEMS_FA3_TLE_AUTO_SPLITKV=1       # experimental metadata-driven split-KV
FLAG_GEMS_FA3_TLE_PAGED_PREFILL_ROUTE=auto|direct|long
FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q=1024
FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_AVG_Q=128
```

`PAGED_PREFILL_ROUTE=auto` keeps decode and short/medium paged workloads on the
direct path, but routes large paged prefill-like requests to the long persistent
kernel when both thresholds are met.  Use `direct` to reproduce the old route
and `long` to force the candidate route for paged cache-KV requests.

## Archived Experiments

The following decode-only experiments are archived in `bak/`:

- `fa_hopper_decode_onepass.py`
- `fa_hopper_decode_splitkv.py`
- `fa_hopper_decode_paged_lb.py`
- `fa_hopper_decode_seesaw.py`
- `experimental_registry.py`

They remain importable for `scripts/run_fa3_decode_experiments.py`, but are not
re-exported by `kernels.py` and are not selected by default dispatch.
