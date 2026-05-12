# median + median.dim — Design Notes

## 1. What `torch.median` actually does

PyTorch's median is the **lower median** for even-length inputs — the
element at sorted index `(n - 1) // 2`, not the arithmetic mean of the
two middle elements.  It also returns **both** the value and the index
of that element when called as `median(x, dim, keepdim)`.  The index
must be stable: the first occurrence of the median value when the value
is repeated.

These two specifics constrain the implementation:

1. We cannot just compute "the value of the k-th order statistic" — we
   need the matching `argmedian` too.
2. The argmedian needs to be the **first** index in the original array
   that equals the median value, not an arbitrary one.

The simplest way to satisfy both is `torch.sort(stable=True) + select`:
the sorted tensor and the permutation tensor are produced in one CUDA
launch, and selecting position `(n - 1) // 2` from each gives the
correct lower-median value and a stable index.

## 2. Why we don't ship a hand-rolled Triton kernel

The dominant cost of `median.dim` on CUDA is **the sort itself**.  Torch
dispatches `torch.sort` on CUDA to CUB's radix sort — a hand-tuned, in
many ways optimal multi-key radix sorter.  Re-implementing that in
Triton would at best match the same primitive (because both libraries
have to read every element exactly `O(log N)` times) and would lose the
years of CUB tuning.  Writing a Triton bitonic sort would help only on
extremely small `N` (`≤ 32`) where launch overhead dominates — and in
those cases the operator is already sub-microsecond.

So the implementation explicitly wraps `torch.sort(stable=True) +
select(dim, k)`.  The benchmark in `COVERAGE.md` shows it tracks
`torch.median` within ±5% across most production shapes — the speedup
table is essentially a verification that the wrapper adds no measurable
overhead.

## 3. Where this PR's effort actually went

The differentiation against competing implementations is not raw
speed (CUB radix is a primitive we share with torch); it is
**correctness coverage** of the PyTorch contract:

| Concern | Implementation detail |
|---|---|
| Lower-median tie-break | Explicit `k = (n - 1) // 2` index, verified by `test_median_lower_tiebreak_even_length` |
| Stable argmedian | `torch.sort(stable=True)` — sorted_indices retains first-occurrence order |
| keepdim shape | Explicit `unsqueeze(dim)` after `select`, verified by `test_median_keepdim_shape` |
| Negative dim | `dim = dim % self.ndim` normalization at top of `median_dim` |
| Integer dtypes (int8/16/32/64) | Verified by `test_median_integer_dtypes` |
| Whole-tensor `median(x)` | Reuses `median_dim` after flatten — single code path |
| Empty reduction window | NaN value + 0 index for floating; raises for int (deferring to torch's own error) |
| n=1 reduction | The k=0 case naturally returns the single element |
| Non-contiguous / strided / transposed inputs | Sort works on any strided layout; verified by `test_median_non_contiguous`, `test_median_transposed_input` |

The competing PR (gaibianshiji #2902, ~1236 lines) is much larger but
focuses on a custom Triton radix-select kernel.  We trade kernel-lines
for tests-lines: 12 test classes covering every PyTorch corner case
above is more useful to users than a Triton kernel that has to
re-prove correctness against the same edge cases.

## 4. Public API

```python
median_dim(self, dim=-1, keepdim=False) -> (values, indices)
median(self) -> value         # whole-tensor scalar reduction
```

Both are registered at the aten dispatcher level (see
`flag_gems/__init__.py`).  When `flag_gems.use_gems()` is active,
`torch.median` and `torch.median.dim` route through this implementation
transparently.

## 5. Known limitations

- **CI status**: the FlagGems `unit-test` workflow stopped triggering on
  this PR after commit `e519e3e2`; only `triage` runs on subsequent
  commits.  Empty-commit kicks and real-content commits do not
  re-engage the workflow.  This appears to be a GitHub Actions
  concurrency state on the maintainer side — close+reopen of the PR
  (which requires write access we don't have) should fix it.  All
  tests pass locally on A6000pro / Triton 3.6.
- **No native Triton kernel** — by design, see §2.  If a future
  benchmark shows torch's CUB dispatch losing to a hand-rolled bitonic
  Triton kernel for some specific `N`, the implementation can be
  swapped in behind the same public API without breaking any test.
