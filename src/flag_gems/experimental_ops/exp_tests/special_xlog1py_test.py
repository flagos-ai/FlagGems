# SPECIAL_XLOG1PY operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.special_xlog1py import (
    special_xlog1py as gems_special_xlog1py,
)
from flag_gems.experimental_ops.special_xlog1py import (
    special_xlog1py_out as gems_special_xlog1py_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_xlog1py_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.3
    ref_self = self.clone()
    ref_other = other.clone()
    ref_out = torch.ops.aten.special_xlog1py(ref_self, ref_other)
    with flag_gems.use_gems():
        act_out = gems_special_xlog1py(self, other)
    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other_scalar", [-0.25, 0.0, 0.5, 1.25])
def test_special_xlog1py_other_scalar(shape, dtype, other_scalar):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_self = self.clone()
    ref_out = torch.ops.aten.special_xlog1py.other_scalar(ref_self, other_scalar)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_xlog1py.other_scalar(self, other_scalar)
    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("self_scalar", [-2.0, -0.5, 0.0, 2.0])
def test_special_xlog1py_self_scalar(shape, dtype, self_scalar):
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.3
    ref_other = other.clone()
    ref_out = torch.ops.aten.special_xlog1py.self_scalar(self_scalar, ref_other)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_xlog1py.self_scalar(self_scalar, other)
    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_xlog1py_out_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.3
    ref_self = self.clone()
    ref_other = other.clone()
    ref_out = torch.empty_like(ref_self)
    torch.ops.aten.special_xlog1py.out(ref_self, ref_other, out=ref_out)
    with flag_gems.use_gems():
        act_self = self.clone()
        act_other = other.clone()
        act_out = torch.empty_like(act_self)
        gems_special_xlog1py_out(act_self, act_other, act_out)
    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("self_scalar", [-2.0, -0.5, 0.0, 2.0])
def test_special_xlog1py_self_scalar_out(shape, dtype, self_scalar):
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.3
    ref_other = other.clone()
    ref_out = torch.empty_like(ref_other)
    torch.ops.aten.special_xlog1py.self_scalar_out(self_scalar, ref_other, out=ref_out)
    with flag_gems.use_gems():
        act_other = other.clone()
        act_out = torch.empty_like(act_other)
        torch.ops.aten.special_xlog1py.self_scalar_out(
            self_scalar, act_other, out=act_out
        )
    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other_scalar", [-0.25, 0.0, 0.5, 1.25])
def test_special_xlog1py_other_scalar_out(shape, dtype, other_scalar):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_self = self.clone()
    ref_out = torch.empty_like(ref_self)
    torch.ops.aten.special_xlog1py.other_scalar_out(ref_self, other_scalar, out=ref_out)
    with flag_gems.use_gems():
        act_self = self.clone()
        act_out = torch.empty_like(act_self)
        torch.ops.aten.special_xlog1py.other_scalar_out(
            act_self, other_scalar, out=act_out
        )
    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=False)


@pytest.mark.special_xlog1py
def test_perf_aten_special_xlog1py():
    # Define input generation logic matching the operator arguments
    def special_xlog1py_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device) - 0.3
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=special_xlog1py_input_fn,
        op_name="special_xlog1py",
        torch_op=torch.ops.aten.special_xlog1py,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
