# [FlagGems Operator Development Competition] Add ctc_loss operator

## Device
Iluvatar BI-V150

## PR Category
- Operator
- OP Test
- Benchmark

## Type of Change
- New Feature

## Description
Add FlagGems ctc_loss operator with full Triton kernel implementation, including forward and backward paths.

### Operator Implementation (`src/flag_gems/ops/ctc_loss.py`, 1336 lines)
- **Forward path**: 3 Triton kernels
  - `_ctc_loss_forward_kernel`: Full forward with alpha table storage for autograd backward.
  - `_ctc_loss_forward_scratch_kernel`: Double-buffered forward using only 2×S scratch space for no-grad/no-reduce paths.
  - `_ctc_loss_forward_reduce_serial_kernel`: Serial reduce kernel for small (batch≤8, T≤128) mean/sum reduction, using single-batch scratch.
- **Backward path**: 2 Triton kernels
  - `_ctc_loss_init_grad_kernel`: Initialize gradient tensor from reduction mode and per-batch neg_log_likelihood.
  - `_ctc_loss_backward_kernel`: Beta-recursion backward kernel computing gradients w.r.t. log_probs.
- **Supported target formats**: Padded (N, S) and concatenated (sum(target_lengths),) targets.
- **Reduction modes**: none (0), mean (1), sum (2).
- **Supported dtypes**: fp32, fp16, bf16 (forward-only path keeps fp16/bf16; backward casts to fp32 for accuracy).
- **Features**: zero_infinity, full-length optimization (input_lengths==T), variable-length support, unbatched 2D input, non-contiguous input handling, input validation.

### Registration
- Registered both `ctc_loss.IntList` and `ctc_loss.Tensor` ATen overloads via `_FULL_CONFIG`.
- Uses `AUTOGRAD_DISPATCH_KEY` extra dispatch keys for autograd support.
- Updated `register.py` to support 4-tuple config format (`op_name, func, condition, extra_dispatch_keys`).

### Tests (`tests/test_ctc_loss.py`, 655 lines, 57 quick / 180+ full cases)
Coverage includes:
- Core forward+backward matrix tests (padded + concatenated targets × none/mean/sum × fp32/fp16/bf16)
- zero_infinity edge cases
- no-grad forward path (including scratch kernel and reduce-serial kernel paths)
- Unbatched 2D input (T, C)
- Variable input/target lengths (input_lengths != T)
- Large concatenated variable-length regression
- Non-contiguous padded zero_infinity
- Input validation: rank, shape, blank, dtype checks
- Empty/repeated target edge cases

### Benchmark (`benchmark/test_ctc_loss.py`, 134 lines)
- Custom `CtcLossBenchmark` class with `DEFAULT_SHAPES = [(64,4,32,16), (256,16,64,48), (512,32,64,48), (1024,32,128,96)]`
- Padded + concatenated target generation per shape
- Combined forward and backward benchmarks (`is_backward=True`)
- Supports fp32/fp16/bf16 dtypes
- Shape description: "T, N, C, S" in `core_shapes.yaml`

## Progress
- [x] Change is fully covered by UT (57 quick / 180+ full test cases)
- [x] Change is properly reviewed (1 reviewer required, 2 recommended)
- [x] Change is part of FlagGems Operator Development Competition

## Tests
```bash
# Quick mode
pytest tests/test_ctc_loss.py -m ctc_loss --quick -s

# Full mode
pytest tests/test_ctc_loss.py -m ctc_loss -s
```

## Performance
```bash
# Core benchmark
pytest benchmark/test_ctc_loss.py --level core --mode kernel

# Comprehensive benchmark
pytest benchmark/test_ctc_loss.py --level comprehensive --mode kernel
```

## Files Changed
| File | Status | Lines |
|------|--------|-------|
| `src/flag_gems/ops/ctc_loss.py` | Added | +1336 |
| `tests/test_ctc_loss.py` | Added | +655 |
| `benchmark/test_ctc_loss.py` | Added | +134 |
| `src/flag_gems/__init__.py` | Modified | +3 |
| `src/flag_gems/ops/__init__.py` | Modified | +2 |
| `src/flag_gems/runtime/register.py` | Modified | +27/-18 |
| `benchmark/core_shapes.yaml` | Modified | +16 |

## Registration Notes
`ctc_loss.IntList` and `ctc_loss.Tensor` are both registered because they correspond to two PyTorch ATen overload schemas for the same logical operator. The overload selected by PyTorch depends on whether `input_lengths` / `target_lengths` are passed as Python lists or tensors. FlagGems uses one shared `ctc_loss` implementation for both overloads, normalizing length inputs internally.

## Notes / Disclosures
- PyTorch is used only as the test reference and benchmark baseline; the operator implementation is fully independent Triton kernels.
- The forward-only path uses double-buffered scratch (2×S per batch) instead of full T×S alpha storage, enabling efficient no-grad execution.
- Backward path stores full alpha table (N×T×(2S+1)) as required by the CTC beta-recursion backward algorithm.
- fp16/bf16 forward path keeps input dtype and promotes loads to fp32 inside Triton to avoid a full tensor cast.
