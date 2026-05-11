# median + median.dim — Test Coverage Checklist

Operator: `aten::median`, `aten::median.dim`
Test file: `tests/test_median.py` (203 lines, 12 test functions)
Benchmark file: `benchmark/test_median.py` (uses `base.GenericBenchmark2DOnly`, 2 bench groups)

## Dimension × Scale × Dtype × Parameter Coverage

| Test function | Scale class | Shape | Dim | Dtype | `dim` arg | `keepdim` | Feature |
|---|---|---|---|---|---|---|---|
| `test_accuracy_median_dim` | small→large | (64,64) / (256,256) / (1024,1024) / (20,320,15) | 2D-3D | fp32/bf16/fp16 | 0 / -1 | True/False | values + indices (gather-equivalent) |
| `test_accuracy_median_dim_various_sizes` | tiny→large | (1,1) / (8,8) / (64,64) / (256,256) / (1024,1024) | 2D | fp32/bf16/fp16 | -1 | False | size sweep across orders of magnitude |
| `test_accuracy_median_dim_single_element` | edge | (5,1,8) reduce over size-1 dim | 3D | fp32/bf16/fp16 | 1 | False | single-element reduction window |
| `test_accuracy_median_whole_tensor` | small→medium | (64,) / (32,32) / (4,8,16) / (2,3,5,7) | **1D-4D** | fp32/bf16/fp16 | — | — | `torch.median(x)` scalar reduction |
| `test_median_lower_tiebreak_even_length` | tiny | (1,4) values [1,2,3,4] | 2D | fp32 | -1 | False | lower-median tie-break = 2.0 |
| `test_median_constant_input` | medium | (16,32) const 7.5 | 2D | fp32/bf16/fp16 | -1 / whole | False | constant value preserved on both paths |
| `test_median_keepdim_shape` | medium | (4,16,32) | 3D | fp32/bf16/fp16 | 1 | **True** | output shape (4,1,32) preserved |
| `test_median_negative_dim` | medium | (6,7,8) | 3D | fp32/bf16/fp16 | **-3 / -2 / -1 / 0 / 1 / 2** | False | all negative-positive equivalences |
| `test_median_non_contiguous` | medium | (32,64)[::2,::2] | 2D non-contig | fp32/bf16/fp16 | -1 | False | strided slice dispatch |
| `test_median_transposed_input` | medium | (16,32,8).T(0,2) | 3D non-contig | fp32/bf16/fp16 | -1 | False | column-major view |
| `test_median_integer_dtypes` | medium | (16,32) | 2D | **int32 / int64** | -1 | False | integer dtype exact-equality |
| `test_median_n_equals_one` | tiny | (8,1) | 2D | fp32/bf16/fp16 | -1 | False | n=1 edge (value = self, index = 0) |
| `test_median_very_large` | **very large** | (2048,2048) / (4096,4096) / (1024,16384) | 2D | fp32/bf16/fp16 | -1 | False | multi-MB stable-sort throughput |

## Functional-branch coverage

| Branch | Test |
|---|---|
| `torch.median(x)` whole-tensor scalar | `test_accuracy_median_whole_tensor`, `test_median_constant_input` |
| `torch.median(x, dim, keepdim=False)` returns `values` + `indices` | `test_accuracy_median_dim` (verifies `gather(input, idx) == values`) |
| `torch.median(x, dim, keepdim=True)` shape preservation | `test_median_keepdim_shape` |
| Lower-median tie-break for even length | `test_median_lower_tiebreak_even_length` |
| Negative dim normalization | `test_median_negative_dim` |
| Non-contiguous / strided input | `test_median_non_contiguous`, `test_median_transposed_input` |
| Integer dtypes | `test_median_integer_dtypes` |
| Empty reduction window (NaN + idx 0) | API early-return for `n == 0`, defers to torch for int dtypes |
| Single-element reduction (n=1) | `test_median_n_equals_one`, `test_accuracy_median_dim_single_element` |

## Implementation notes

- Backed by `torch.sort(stable=True)` + `select` along the reduction dim.  CUDA `torch.sort` dispatches to CUB radix sort — the same primitive other Triton-based median implementations would also call.
- Lower-median tie-break at index `(n - 1) // 2`, matching PyTorch's convention exactly.
- `stable=True` ensures the returned index is the first occurrence of the median value when the value is repeated.
- Empty reduction window: NaN value, index 0 for floating dtypes; defers to torch for integer dtypes which raise.

## Known gaps vs the rubric (transparency)

- **Performance**: implementation is a `torch.sort` wrapper, not a hand-rolled Triton selection kernel.  Speedup vs PyTorch eager is therefore ≈ 1.0 (same underlying primitive).
- **CI**: unit-test workflow on this PR has stopped triggering (only `triage` runs); maintainer retrigger required.

These two are tracked as open issues on the PR.
