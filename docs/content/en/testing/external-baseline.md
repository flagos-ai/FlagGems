---
title: External NVIDIA performance baselines
weight: 35
---

# External NVIDIA performance baselines

Pass `--base-data` to compare candidate timings with an `ops-baseline` JSON file:

```bash
pytest benchmark/test_add.py --level core --record json \
  --base-data=/absolute/path/pytorch_baseline.json

python tools/run_tests.py --ops add --gpus 0 \
  --base-data /absolute/path/pytorch_baseline.json
```

Use the `--base-data=...` form with pytest, particularly for files outside the
checkout, so early pytest root-directory discovery does not interpret the file
as a test path. The batch runner resolves its path before starting workers.
The filename is arbitrary; the file must use the `ops-baseline` layout:

```json
{
  "neg": {
    "shapes": {
      "[64, 64]": {
        "torch.float32": {"latency_ms": 0.01}
      }
    }
  }
}
```

This example illustrates the format; its latency is synthetic. `T_us` is also
accepted, converted to milliseconds when `latency_ms` is absent. Nonpositive,
nonfinite, boolean, or nonnumeric latencies are not used. Missing/unreadable or
invalid JSON files produce a warning and disable external comparison.

## Matching

Operator names and dtype keys must match exactly, including leading underscores.
For records with `config.inputs.args` and `config.inputs.kwargs`, the matcher
checks nested input structure, scalar values/types, and recorded tensor shape,
dtype, stride, and `requires_grad`. This supports semantic shape keys and the
PyTorch baseline modules from ops-baseline PR #8. Explicit defaults and omitted
defaults in an args/kwargs record are different calls for matching purposes.

The benchmark passes its chosen execution device to the matcher. A named
`device` parameter recorded as `cuda` or `cuda:0` can match that chosen
accelerator (for example `npu` or `mlu:2`). CPU placement, a non-default source
CUDA index, other parameter names, and devices other than the chosen target
remain significant. This does not discard tensor layout or auxiliary-device
information. `torch.device` arguments are described as device strings.

Named `config.inputs` records can match when the benchmark's reference callable
has an inspectable Python signature. Defaults are bound before comparison;
the metadata must describe every bound parameter, including defaults. Empty or
partial named metadata is left unmatched because omitted parameters can change
semantics. Unavailable parameter binding also leaves the record unmatched.

Legacy records without input metadata are accepted only for a single direct
tensor argument with no keywords, using its exact shape key and dtype. Multiple
tensors, scalar parameters, and nested inputs require full input metadata. Even
a unary legacy record cannot verify layout or defaults; supply input metadata
when these affect the workload. There are no implicit operator aliases. Multiple matching records are ambiguous and are not
used. An unmatched record retains native behavior and records `base_miss_reason`.

## Results and hardware scaling

The ordinary Torch latency, Gems latency, and native `speedup` remain measured
as before. A match adds a `vs Base` column and `compared_speedup` to the benchmark
JSON. Its raw value is:

```text
raw ratio = external NVIDIA latency / current Gems latency
compared_speedup = raw ratio / scaling factor
```

If `_scaling_factors.chips` provides a matching vendor/chip, the factor uses the
record's `bottle_neck_unit`: `mem` selects bandwidth, `cuda` selects vector
compute, and `tensor` selects Tensor Core compute for the workload dtype.
Measured values take precedence over nominal values. Missing/invalid factors
leave the raw ratio unchanged. A known chip name must match; multiple matching
models or an unknown model with multiple vendor entries do not select a factor.
Chip matching uses model boundaries, so H20 does not match H200; suffixes such
as H20-3e are accepted. The explicit vendor alias `t-head` / `thead` bridges
ops-baseline's spelling and the FlagGems runtime name.
The configured reference chip must correspond to the baseline collection;
normalization is not proof of equivalent numerical semantics or equal hardware.

The JSON records source path, matched shape key, external latency, raw ratio,
and the selected factor (null if unavailable). `tools/run_tests.py` uses
`compared_speedup` as the final per-case `speedup` and in its existing arithmetic
mean summary, falling back to native speedup on each miss. Summary details retain
`native_speedup` on matches and the external provenance fields. Native-only runs
keep their original columns and aggregation behavior.

`--base-data` does not disable native execution. Existing `skip_native` markers
remain responsible for missing vendor baselines; those cases can still receive
an external speedup. If neither baseline is available, native speedup stays N/A.
The option does not change correctness, preflight, reference-only, profiling,
or timing input generation, and it does not download data from KS3.

BLAS serial runs and parallel workers use the same comparison logic. Workers
receive the absolute baseline path; paired native/Gems inputs are matched using
the native callable ABI. The standalone int4 packing benchmark runners do not
yet apply external baselines. This option does not change custom runners’
existing support for other options such as `--skip-native`.
