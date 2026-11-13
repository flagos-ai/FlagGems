# Independent Hygon FlashAttention performance verification

Measured 2026-11-12 on Hygon gfx936. Official-master baseline: `9a9aecb53103cc83dda27bc7458fc3bfd07f3188`. Isolated candidate: `efd32566fb2fd258f23d6dea469d39088a929381`.

All four declared BF16 cases passed **complete output-tensor and LSE reference comparison, 50 samples per provider, and independent A/A with both block drifts <=5%**. All four meet native 90%.

| Case | Old FG (us) | New FG (us) | Native 2.8.4 (us) | Old/New | Native/New | Correctness/stability |
|---|---:|---:|---:|---:|---:|---|
| flash-model_state | 4670.6409 | 1742.1029 | 1611.3790 | 2.6810x | 0.9250 | pass |
| flash-expanded_small | 9.2880 | 5.5600 | 7.2880 | 1.6705x | 1.3108 | pass |
| flash-expanded_medium | 10.2400 | 6.3160 | 8.3040 | 1.6213x | 1.3148 | pass |
| flash-expanded_large | 76.9279 | 44.5800 | 41.7840 | 1.7256x | 0.9373 | pass |

The STATE case has shared Q/K/V backing and offsets0/2048/4096; the three expanded cases use the independently reconstructed backings specified by their metadata. Values are CPU-direct BF16 random draws from frozen seeds. This validates the disclosed reconstructed contracts; it does not claim replay of original model tensor payloads.

## Comparator identity and accounting

Native comparator: `2.8.4+dtk2604.torch2100.2609292239.g74de14`, compatible with the Torch2.10 + Torch-FL R7 test environment. Every process asserts the actual imported extension origin, metadata version and SO hash **before any workload**; profiled workers check the hash again at termination. The extension hash is `4dbb514cd0d6b30b720bbc739b13f3655ec5b345765b303e463184a1cd243f25`; wheel hash `c2cd914032b188452166661537bf89496810e49e658ebcd1a74e36624e6afadb`. No shared installation is changed.

The `das` role in the machine-readable evidence denotes this fixed native FlashAttention comparator, called through `flash_attn_2_cuda.fwd` with zero-copy native views and normalized output/LSE. It does not claim that shared default Torch-FL boxing was upgraded. FG roles invoke the public ATen overload through exact process-private PrivateUse1 bindings.

Each of old FG, new FG, native and their independent repeats has two blocks of25 ROIs, twenty calls per ROI. Device timing sums all producer-linked kernel durations and divides by twenty. Atomic gates, compile/autotune, output transfers and API costs are outside this metric. No events, ROIs or samples are omitted or pooled.

`cases.csv`/`cases.json` retain full geometry, aliases, offsets, dtypes, seeds and qualification flags. `measurements.json` retains all50 samples per role and source/schema/backing/shard hashes. `kernel-witness.json` lists actual device activities and strict accounting-file fingerprints. Full raw profiler binaries are outside the repository.

```bash
pytest tests/test_flash_attention.py -q --ref cpu
python docs/hygon-optimization/flash-attention/performance-20261112/verify_results.py
```

## Bottom-layer execution path

FG data moves through Torch-FL into `flagos` tensors; the public ATen overload enters the exact PrivateUse1 binding and Hygon provider. The guarded implementation loads Q/K/V using the preserved strides/offsets and launches the compiled Hygon attention kernel on the current stream. Native roles use zero-copy CUDA views of the same backing and the pinned2.8.4 extension. Both providers are checked against blocked CPU FP32 softmax/output and logsumexp references. No Dynamo/Inductor graph fusion is active.
