# Overview
The `experimental_ops` module provides a space for new operators that are not yet ready for production release. Operators in this module are accessible via `flag_gems.experimental_ops.*` and follow the same development patterns as core operators.

# Usage Example
Users can access operators as:
```
import flag_gems

# Global enablement
flag_gems.enable()
result = flag_gems.experimental_ops.your_operator(*args)

# Or scoped usage
with flag_gems.use_gems():
    result = flag_gems.experimental_ops.your_operator(*args)
```


# File Structure
```
src/flag_gems/experimental_ops/
├── __init__.py                 # Module initialization
├── rmsnorm.py          # Example operator implementation
├── [other_operators].py   # Additional operators
├── exp_tests/                 # Accuracy test and performance test
    ├── __init__.py
    ├── rmsnorm_test.py
    ├── [other_operators]_test.py
```

# Operator List
Total: 122 operators
| Op name | Speedup vs CUDA |
|---|---:|
| _safe_softmax | 7.344881 |
| hinge_embedding_loss | 4.040925402 |
| soft_margin_loss | 3.368846639 |
| margin_ranking_loss | 2.246512241 |
| masked_scatter | 2.141666824 |
| zero | 1.738164 |
| special_i0e | 1.462517868 |
| eye | 1.374681 |
| t_copy | 1.364235 |
| prelu | 1.322912173 |
| trace | 1.303166 |
| _upsample_nearest_exact1d | 1.297378 |
| diag | 1.256865 |
| lift_fresh_copy | 1.211434 |
| i0_ | 1.193042 |
| arcsinh | 1.184800587 |
| i0 | 1.18358 |
| replication_pad3d | 1.173729 |
| upsample_nearest1d | 1.172469 |
| tril | 1.141599 |
| pixel_unshuffle | 1.128692 |
| glu | 1.105081 |
| reflection_pad1d | 1.101862 |
| rrelu_with_noise_backward | 1.100563 |
| special_i1 | 1.083081 |
| silu_ | 1.077688 |
| logical_xor_ | 1.076979912 |
| logit | 1.069491804 |
| selu_ | 1.067362 |
| alias_copy | 1.05612 |
| upsample_nearest3d | 1.055983 |
| selu | 1.053783 |
| logit_ | 1.049731 |
| copy_ | 1.049225518 |
| clamp_max_ | 1.047113718 |
| relu6 | 1.043199 |
| arctanh_ | 1.038952 |
| digamma_ | 1.037068 |
| _functional_sym_constrain_range_for_size | 1.035281278 |
| sinh_ | 1.034496 |
| hardsigmoid | 1.034157 |
| asinh_ | 1.034051 |
| sgn_ | 1.03034 |
| hardswish_ | 1.029559 |
| log1p_ | 1.029292718 |
| replication_pad1d | 1.027806 |
| logaddexp | 1.026881514 |
| arcsinh_ | 1.026679 |
| fmin | 1.025362775 |
| hardswish_backward | 1.024932631 |
| softshrink | 1.024263293 |
| softplus | 1.022865 |
| floor_ | 1.022106 |
| sigmoid_ | 1.020004 |
| hypot | 1.017964027 |
| absolute | 1.017875 |
| huber_loss | 1.01661664 |
| neg_ | 1.015127 |
| hardsigmoid_ | 1.014869 |
| absolute_ | 1.014659 |
| rad2deg_ | 1.014513 |
| exp2_ | 1.013567 |
| greater_equal_ | 1.012778286 |
| smooth_l1_loss | 1.012750638 |
| hardtanh_ | 1.012527 |
| atanh_ | 1.012010964 |
| ge_ | 1.011516781 |
| eq_ | 1.011346249 |
| lt_ | 1.011211413 |
| logaddexp2 | 1.01048221 |
| hypot_ | 1.010429673 |
| leaky_relu_ | 1.010058963 |
| not_equal_ | 1.009854358 |
| le_ | 1.009851716 |
| ne_ | 1.009832832 |
| xlogy_ | 1.009714773 |
| negative | 1.009226 |
| greater_ | 1.008626951 |
| addcmul_ | 1.008597457 |
| less_ | 1.008430982 |
| multiply | 1.007042884 |
| xlogy | 1.006288915 |
| heaviside_ | 1.006215472 |
| exp2 | 1.005911 |
| leaky_relu | 1.005823175 |
| heaviside | 1.004409567 |
| scalar_tensor | 1.003317098 |
| sigmoid | 1.002895 |
| sign | 1.002807 |
| slice_scatter | 1.001129 |
| log2_ | 1.000691305 |
| deg2rad | 1.000294412 |
| empty_permuted | 1.000243665 |
| hardtanh | 0.997583 |
| square | 0.996088551 |
| deg2rad_ | 0.995312 |
| addcdiv | 0.994735 |
| negative_ | 0.994018943 |
| trunc | 0.993915 |
| new_ones | 0.989766621 |
| fix_ | 0.98693 |
| exp_ | 0.98398 |
| hardshrink | 0.982452266 |
| cos_ | 0.98114 |
| sinc_ | 0.976825 |
| frac | 0.975216 |
| cosh_ | 0.973754 |
| fix | 0.969886 |
| log_ | 0.960865 |
| reciprocal_ | 0.958989 |
| sin_ | 0.957053 |
| rsqrt_ | 0.95315 |
| pixel_shuffle | 0.936782 |
| arccosh | 0.913407 |
| arctanh | 0.907158761 |
| replication_pad2d | 0.888932 |
| sgn | 0.871724 |
| take | 0.854309 |
| erfinv | 0.853474081 |
| hardswish | 0.848665 |
| _log_softmax_backward_data | 0.820464 |
| less_equal_ | 0.800514773 |

# Adding New Operators
## 1. Create Operator Implementation
Create your operator file in `src/flag_gems/experimental_ops/`:
```
# src/flag_gems/experimental_ops/your_operator.py
from flag_gems.utils import libentry

@libentry()
@triton.autotune(
    configs=[...],
    key=[...]
)
def your_operator_kernel(...):
    # Triton kernel implementation
    pass

def your_operator(*args, **kwargs):
    # Python wrapper
    return your_operator_kernel(*args, **kwargs)
```

## 2. Update Module Exports
Add your operator to `src/flag_gems/experimental_ops/__init__.py` :
```
from .your_operator import your_operator
__all__ = ["rmsnorm", "your_operator"]
```

## 3. Update Main Module
The experimental_ops module is already integrated in the main `__init__.py` . No changes needed there.


# Testing
## Accuracy Tests
Add accuracy test in `exp_tests/your_ops_test.py`:
```
import pytest
import torch
import flag_gems
from tests.accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
    to_reference,
)

@pytest.mark.your_operator
@pytest.mark.parametrize("shape", [...])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_your_operator(shape, dtype):
    # Test implementation
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    # Reference implementation
    ref_out = torch.your_operator(ref_inp, ...)

    # FlagGems implementation
    with flag_gems.use_gems():
        res_out = flag_gems.experimental_ops.your_operator(inp, ...)

    gems_assert_close(res_out, ref_out, dtype)
```

## Performance Tests
Add performance test in `exp_tests/your_ops_test.py`:
```
import pytest
import torch
import time
import flag_gems

class TestYourOperatorPerf:
    def setup_method(self):
        flag_gems.enable()

    def teardown_method(self):
        flag_gems.disable()

    @pytest.mark.your_operator
    @pytest.mark.parametrize("shape", [...])
    def test_perf_your_operator(self, shape):
        inp = torch.randn(shape, device=flag_gems.device)

        # Warmup
        for _ in range(10):
            _ = flag_gems.experimental_ops.your_operator(inp)

        torch.cuda.synchronize()

        # Benchmark FlagGems
        start_time = time.time()
        for _ in range(100):
            out = flag_gems.experimental_ops.your_operator(inp)
        torch.cuda.synchronize()
        gems_time = (time.time() - start_time) / 100

        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(100):
            ref_out = torch.your_operator(inp)
        torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / 100

        speedup = torch_time / gems_time
        print(f"YourOperator {shape}: Speedup {speedup:.2f}x")

        assert speedup > 1.0, "Should be faster than PyTorch"
```

# CI Integration
Add tests ad performace tests to the CI workflow `.github/workflows/gems-experimental-test.yaml` .
