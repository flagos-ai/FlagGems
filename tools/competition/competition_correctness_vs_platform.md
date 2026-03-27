# Competition Correctness: Current Implementation vs Platform Requirement (4.1.1)

## Purpose

This document summarizes the differences between:

- Platform requirement: **Section 4.1.1 功能正确性**
- Current FlagGems correctness checking used by competition/unit tests

It is intended for reviewers and operator owners to quickly understand which parts already align, which parts differ (especially `rtol/atol`), and what changes would be required to fully match the platform spec.

## Platform requirement (4.1.1) summary

### Correctness expectations

- Operator outputs must align with the official baseline.
- Should cover boundary cases (extremes, zeros, negatives, empty tensors, dynamic shapes).
- Invalid inputs (illegal dtype, dimension mismatch) should have clear error-handling behavior.

### Numerical comparison rule

Platform describes using `torch.allclose(input, other, rtol, atol)`:

- Condition:

  `|input_i - other_i| <= atol + rtol * |other_i|`

- If an operator (or dtype) can be bitwise identical to the reference:
  - `rtol = 0`, `atol = 0`
- Otherwise:
  - `rtol = 1e-4`
  - `atol` is dtype-dependent (table provided), e.g.
    - `float16: 1e-3`
    - `float32: 1.3e-6`
    - `bfloat16: 0.016`
    - integer/bool: `0`

## Current FlagGems implementation

### Where comparisons happen

Most tests use helpers in `tests/accuracy_utils.py`:

- `gems_assert_close(...)`
- `gems_assert_equal(...)`

These call into `src/flag_gems/testing/__init__.py`:

- `assert_close(res, ref, dtype, ...)`
- `assert_equal(res, ref, ...)`

### Current thresholds

#### `assert_equal`

Implementation:

- `torch.testing.assert_close(res, ref, atol=0, rtol=0)`

This matches the platform’s “bitwise identical” requirement *when tests explicitly choose to use it*.

#### `assert_close`

Implementation (simplified):

- `rtol = RESOLUTION[dtype]`
- `atol = atol_argument * reduce_dim`
- `torch.testing.assert_close(res, ref, atol=..., rtol=...)`

Defaults:

- `atol_argument = 1e-4`
- `reduce_dim = 1`

`RESOLUTION` is a dtype-keyed table defined in the same module. For non-kunlunxin vendors it includes:

- `bool/int*: 0`
- `float8*: 1e-3`
- `float16: 1e-3`
- `float32: 1.3e-6`
- `bfloat16: 0.016`
- `float64: 1e-7`
- `complex32: 1e-3`
- `complex64: 1.3e-6`

## Differences vs platform requirement

### 1) `rtol` policy

- Platform (non-bitwise): fixed `rtol = 1e-4`.
- Current: `rtol` is dtype-dependent via `RESOLUTION`.

Impact:

- `float16`: current `rtol=1e-3` is **looser** than platform `1e-4`.
- `bfloat16`: current `rtol=0.016` is **much looser** than platform `1e-4`.

### 2) `atol` policy

- Platform: dtype-dependent `atol` table.
- Current: default `atol_argument=1e-4` (constant) and scales by `reduce_dim`.

Impact:

- `float16`: current default `atol=1e-4` is **stricter** than platform `1e-3`.
- `bfloat16`: current default `atol=1e-4` is **much stricter** than platform `0.016`.

This can cause failures where abs error is small but greater than `1e-4` (e.g. some fp16 reduction-like cases).

### 3) `reduce_dim` scaling is project-specific

- Platform spec does not mention scaling tolerance by reduction size.
- Current multiplies `atol` by `reduce_dim` (only if callers pass a value > 1).

### 4) Bitwise rule is manual (test-author choice)

- Platform: if bitwise is achievable, `rtol=0, atol=0`.
- Current: only enforced if tests use `assert_equal` / `gems_assert_equal`.

### 5) Boundary/invalid input coverage is not enforced automatically

- Platform explicitly requires boundary and invalid-input behavior.
- Current competition correctness tests are mostly happy-path numerical comparisons.

Full compliance would require adding explicit tests for:

- Empty tensors
- Dynamic shapes
- Illegal dtype
- Dimension mismatch
- Expected error type/message (if required)

## Alignment options

### Option A: Keep current implementation; clarify documentation

- Treat `RESOLUTION` + `atol=1e-4*reduce_dim` as the repo’s authoritative test standard.
- Document that it differs from platform text.

Pros:

- Minimal risk / minimal CI churn

Cons:

- Not strictly aligned with platform requirement text

### Option B: Update implementation to match platform spec

Potential changes:

- Use fixed `rtol=1e-4` (except bitwise cases)
- Use dtype-dependent `atol` table as specified
- Decide whether to keep `reduce_dim` scaling (and how)

Pros:

- Matches platform requirement text closely

Cons:

- Likely causes widespread threshold changes; needs validation across vendors/backends

## References

- Platform requirement: Section 4.1.1 功能正确性 (provided in review context)
- Current code:
  - `src/flag_gems/testing/__init__.py`
  - `tests/accuracy_utils.py`
  - `tests/test_competition_ops.py`
