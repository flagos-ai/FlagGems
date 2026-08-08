import pytest
import torch

import flag_gems

from . import base

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

if VENDOR == "nvidia":
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]

# pivot=False is only supported on CUDA
if DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]

_CHECK_ERRORS_VALUES = [False, True]

# Use the same shapes as linalg_lu_factor for consistency
LU_FACTOR_EX_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 512),
    (32, 16),
    (16, 32),
    (128, 64),
    (64, 128),
    (4, 32, 32),
    (128, 16, 16),
    (1024, 512, 512),
    (4096, 512, 512),
]


class LinalgLuFactorExBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot, check_errors"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_FACTOR_EX_SHAPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                # Only test check_errors variations for pivot=True to keep
                # the benchmark size manageable.
                check_errors_list = _CHECK_ERRORS_VALUES if pivot else [False]
                for check_errors in check_errors_list:
                    inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                    yield inp, {"pivot": pivot, "check_errors": check_errors}


@pytest.mark.linalg_lu_factor_ex
def test_linalg_lu_factor_ex():
    bench = LinalgLuFactorExBenchmark(
        op_name="linalg_lu_factor_ex",
        torch_op=torch.linalg.lu_factor_ex,
        gems_op=flag_gems.linalg_lu_factor_ex,
        dtypes=_TEST_DTYPES,
    )
    bench.run()


class LinalgLuFactorExOutBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape, pivot, check_errors"
    DEFAULT_DTYPES = _TEST_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_FACTOR_EX_SHAPES

    def get_input_iter(self, dtype):
        for inp_shape in self.shapes:
            inp_shape = tuple(inp_shape)
            for pivot in _PIVOT_VALUES:
                if pivot:
                    continue  # out variant already covered by main benchmark
                k = min(inp_shape[-2], inp_shape[-1])
                batch_shape = inp_shape[:-2]
                inp = torch.randn(inp_shape, dtype=dtype, device=self.device)
                LU = torch.empty(inp.shape, dtype=dtype, device=inp.device)
                pivots = torch.empty(
                    (*batch_shape, k), dtype=torch.int32, device=inp.device
                )
                info = torch.empty(batch_shape, dtype=torch.int32, device=inp.device)
                yield inp, {"out": (LU, pivots, info)}


@pytest.mark.linalg_lu_factor_ex_out
def test_linalg_lu_factor_ex_out():
    bench = LinalgLuFactorExOutBenchmark(
        op_name="linalg_lu_factor_ex_out",
        torch_op=torch.linalg.lu_factor_ex,
        gems_op=flag_gems.linalg_lu_factor_ex_out,
        dtypes=_TEST_DTYPES,
    )
    bench.run()
