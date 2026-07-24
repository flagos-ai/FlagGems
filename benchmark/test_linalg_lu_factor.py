import pytest
import torch

import flag_gems

from . import base

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if DEVICE == "cuda":
    _TEST_DTYPES = [torch.float32, torch.float64]
elif DEVICE == "npu":
    _TEST_DTYPES = [torch.float32]
else:
    _TEST_DTYPES = [torch.float32, torch.float64]

# pivot=False is only supported on CUDA
if DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]


class LinalgLuFactorBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = _TEST_DTYPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                yield inp, {"pivot": pivot}


@pytest.mark.linalg_lu_factor
def test_linalg_lu_factor():
    bench = LinalgLuFactorBenchmark(
        op_name="linalg_lu_factor",
        torch_op=torch.linalg.lu_factor,
        dtypes=_TEST_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_lu_factor)
    bench.run()
