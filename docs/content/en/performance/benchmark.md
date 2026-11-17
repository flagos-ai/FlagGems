---
title: Performance Benchmark
weight: 20
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# Performance Benchmarking in FlagGems

It is recommended to follow the steps below to add test cases for a new operator.
These steps apply to Python-based operators as well as C++-wrapped operators.

{{% steps %}}

1. **Select the appropriate test file**

   <!--TODO(Qiming): remove the following constraints. -->
   Based on the type of operator, choose the corresponding file in the `benchmark`
   directory:

   - For reduction operators, add the test case to `test_reduction_perf.py`.

   - For tensor constructor operators, add the test case to `test_tensor_constructor_perf.py`.

   - If the operator doesn't fit into an existing category, you can add it to `test_special_perf.py`
     or create a new file for the new operator category.

1. **Check existing benchmark classes**

   Once you've identified the correct file, review the existing classes that inherit
   from the `Benchmark` structure to see if any fit the test scenario for your operator,
   specifically considering:

   - Whether the **metric collection** is suitable.

   - Whether the **input generation function** (`input_generator` or `input_fn`) is appropriate.

1. **Add test cases**

   Depending on the test scenario, follow one of the approaches below to add the test case:

   - **Using existing metric and input generator**

     If the existing metric collection and input generation function meet the requirements of your operator,
     you can add a line of `pytest.mark.parametrize` directly, following the code organization in the file.
     For example, see the operators in `test_binary_pointwise_perf.py`.

   - **Custom input generator**

     If the metric collection is suitable but the input generation function does not meet the operator's requirements,
     you can implement a custom `input_generator`.
     Refer to the `topk_input_fn` function in `test_special_perf.py` as an example of a custom input function
     for the `topk` operator.

   - **Custom metric and input generator**

     If neither the existing metric collection nor the input generation function meets the operator's needs,
     you can create a new class. The new class should define operator-specific metric collection logic
     and a custom input generator. You can refer to various `Benchmark` subclasses across the `benchmark` directory
     for examples.
{{% /steps %}}

## Running benchmarks with model-profiling shapes

FlagGems ships per-model profiling shape files under `benchmark/models_shapes/`.
These YAML files record the tensor shapes observed during real model inference,
along with the GPU time share (% of total phase time and call count) for each
operator.  Using `--model-profile` runs only those shapes and adds a
**Time Share** column to the output so you can prioritise optimisation work.

### Available shape files

| File | Model | Hardware | Collected |
|------|-------|----------|-----------|
| `benchmark/models_shapes/285B.yaml` | 285B MoE (DeepSeek-V3.2 architecture) | 8×H100 80GB, TP=8 | 2026-11-16 |

### Using `--model-profile`

```bash
# Run all operators listed in the 285B profile:
pytest benchmark/test_mm.py benchmark/test_bmm.py \
       benchmark/test_rms_norm.py benchmark/test_add.py \
       benchmark/test_mul.py benchmark/test_copy.py \
       benchmark/test_fill.py benchmark/test_zeros.py \
       benchmark/test_silu.py benchmark/test_silu_and_mul.py \
       benchmark/test_softmax.py \
       --model-profile benchmark/models_shapes/285B.yaml --level core -s

# Or use the helper script that auto-selects markers from the YAML:
python benchmark/benchmark_for_models.py \
       --shape-file benchmark/models_shapes/285B.yaml
```

The `--model-profile` flag does two things:

1. **Overrides `--shape_file`** with the given YAML, so each benchmark runs only
   the shapes captured from that model.  Operators not present in the YAML are
   silently skipped (their existing shapes are unaffected when you run without
   the flag).
2. **Attaches profiling metadata** to every result row.  A `Time Share
   (prefill/decode)` column appears in the terminal output, formatted as:

   ```
   pf 10.99%/35520x | dc 23.86%/84760x
   ```

   where `pf` is the prefill phase and `dc` is the decode phase.

### Shape file format

Each entry in a model shape YAML has three sections:

```yaml
mm:
  profiling:
    prefill:
      pct: 10.99    # % of total prefill GPU time
      calls: 35520  # kernel invocations during profiling window
    decode:
      pct: 23.86
      calls: 84760
  shapes:
    - [1, 8192, 3072, 1536]   # [b, M, N, K] for BlasBenchmark (mm/bmm)
    - [1, 8192, 4096, 2048]
  shape_desc: "b, M, N, K"
```

The `op_name` key (e.g. `mm`) must match the `op_name` passed to the
`Benchmark` subclass for the flag to take effect.  Shape tuple format follows
the convention of the corresponding benchmark class:

| Class | Shape format |
|-------|-------------|
| `BlasBenchmark` (`mm`, `bmm`) | `[b, M, N, K]` |
| `GenericBenchmark2DOnly` (`rms_norm`) | `[M, N]` |
| `BinaryPointwiseBenchmark` / `GenericBenchmark` (`add`, `mul`, `copy_`, `silu_and_mul`) | `[M, N]` |
| `UnaryPointwiseBenchmark` (`silu`, `zeros`) | `[M, N]` |
| `GenericBenchmark` scalar (`fill_scalar`) | `[N]` |
| `UnaryReductionBenchmark` (`softmax`) | `[M, N]` |

### Adding a new model shape file

1. Create `benchmark/models_shapes/<ModelName>.yaml` following the format above.
2. Collect profiling data from a real inference run (e.g. using
   `torch.profiler` with `record_shapes=True`).
3. For each operator that FlagGems supports (registered in
   `src/flag_gems/__init__.py`), add an entry with the observed shapes and
   profiling percentages.
4. Add a row for the new file to the **Available shape files** table above.
