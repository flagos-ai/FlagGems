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


class LinalgLuFactorOutBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot"
    DEFAULT_DTYPES = _TEST_DTYPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                k = min(inp_shape[-2], inp_shape[-1])
                batch_shape = inp_shape[:-2]
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                LU = torch.empty(inp.shape, dtype=dtype, device=inp.device)
                pivots = torch.empty(
                    (*batch_shape, k), dtype=torch.int32, device=inp.device
                )
                yield inp, {"pivot": pivot}, {"out": (LU, pivots)}


@pytest.mark.linalg_lu_factor
def test_linalg_lu_factor_out():
    bench = LinalgLuFactorOutBenchmark(
        op_name="linalg_lu_factor_out",
        torch_op=torch.linalg.lu_factor,
        dtypes=_TEST_DTYPES,
    )
    bench.set_gems(flag_gems.linalg_lu_factor_out)
    bench.run()
