# Overview
The `experimental` module provides a space for new operators that are not yet ready for production release. Operators in this module are accessible via `flag_gems.experimental.*` and follow the same development patterns as core operators.

# Usage Example
Users can access operators as:
```
import flag_gems

# Global enablement
flag_gems.enable()
result = flag_gems.experimental.layer_norm(*args)
result = flag_gems.experimental.your_operator(*args)

# Or scoped usage
with flag_gems.use_gems():
    result = flag_gems.experimental.layer_norm(*args)
    result = flag_gems.experimental.your_operator(*args)
```


# File Structure
```
src/flag_gems/experimental/
├── __init__.py                 # Module initialization
├── layer_norm.py          # Example operator implementation
├── [other_operators].py   # Additional operators
├── exp_tests/                 # Accuracy test and performance test
    ├── __init__.py
    ├── test_layer_norm_accuracy.py
    ├── test_layer_norm_performace.py
```

# Adding New Operators
## 1. Create Operator Implementation
Create your operator file in `src/flag_gems/experimental/`:
```
# src/flag_gems/experimental/your_operator.py
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
Add your operator to `src/flag_gems/experimental/__init__.py` :
```
from .your_operator import your_operator
__all__ = ["layer_norm", "your_operator"]
```

## 3. Update Main Module
The experimental module is already integrated in the main `__init__.py` . No changes needed there.


# Testing
## Accuracy Tests
Add accuracy test in `exp_tests/test_your_ops_accuracy.py` following the pattern:
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
        res_out = flag_gems.experimental.your_operator(inp, ...)

    gems_assert_close(res_out, ref_out, dtype)
```

## Performance Tests
Add performance test in `exp_tests/test_your_ops_performance.py` following the pattern:
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
            _ = flag_gems.experimental.your_operator(inp)

        torch.cuda.synchronize()

        # Benchmark FlagGems
        start_time = time.time()
        for _ in range(100):
            out = flag_gems.experimental.your_operator(inp)
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
Add accuracy tests ad performace tests to the CI workflow `.github/workflows/gems-experimental-test.yaml.yaml` .
