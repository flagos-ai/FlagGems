## Summary

This PR implements the `_sobol_engine_draw` operator for FlagGems using Triton kernels. This operator generates quasi-random Sobol sequence points, which are widely used in Monte Carlo integration, optimization, and quasi-Monte Carlo methods.

## Implementation Details

### Triton Kernels
- **`sobol_draw_kernel_simple`**: Sequential kernel optimized for small workloads (dimension ≤ 32, n ≤ 10000)
- **`sobol_engine_draw_kernel`**: Dimension-blocked kernel for larger workloads, parallelized across dimensions

### Algorithm
The implementation uses the Gray code XOR algorithm:
1. For each sample `i`, find the rightmost zero bit position in `(num_generated + i)`
2. XOR the current quasi state with the direction vector at that position
3. Convert the result to float in range [0, 1)
4. Update quasi state for the next iteration

### Technical Challenges Solved
- **Sequential Dependencies**: Sobol generation requires each sample to depend on the previous quasi state, limiting parallelization. The implementation processes samples sequentially while parallelizing across dimensions for larger workloads.
- **Triton Limitations**: Worked around Triton's lack of `break` statement support by using conditional flags to track when the rightmost zero bit is found.
- **Power-of-2 Requirements**: Used `triton.next_power_of_2()` with masking to handle arbitrary dimension sizes with `tl.arange`.

## Testing

All 29 tests pass successfully:
- ✅ 24 parameterized accuracy tests covering various dimensions (2, 5, 10, 20), sample counts (10, 100, 1000), and dtypes (float32, float64)
- ✅ 3 tests with non-zero offsets (0, 10, 100)
- ✅ 1 edge case test (dimension=1, n=1)
- ✅ 1 large workload test (n=500, dimension=50)

Results match PyTorch CPU reference exactly with `torch.allclose(rtol=1e-5, atol=1e-8)`.

## Files Changed

### New Files
- `src/flag_gems/ops/underscore_sobol_engine_draw.py` - Triton kernel implementation
- `tests/test_underscore_sobol_engine_draw.py` - Comprehensive test suite
- `benchmark/test_underscore_sobol_engine_draw.py` - Performance benchmark configuration

### Modified Files
- `src/flag_gems/ops/__init__.py` - Added import and `__all__` export
- `src/flag_gems/__init__.py` - Registered operator in aten interface
- `conf/operators.yaml` - Added operator metadata

## Notes

- PyTorch's CUDA implementation of `_sobol_engine_draw` has a segfault bug in version 2.12.0+cu130, making this FlagGems implementation particularly valuable as a working CUDA alternative.
- The sequential nature of Sobol generation limits parallelization, but the implementation is efficient for typical use cases.
- Performance benchmarks are configured for workload sizes from (100, 2) to (100000, 3).

## Checklist

- [x] Implementation follows FlagGems code style and conventions
- [x] All pre-commit hooks pass (flake8, black, isort, etc.)
- [x] Comprehensive test suite added with 100% pass rate
- [x] Performance benchmark configuration added
- [x] Operator registered in FlagGems interface
- [x] Operator metadata added to operators.yaml
- [x] Results match PyTorch CPU reference exactly
