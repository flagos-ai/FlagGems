# COPY_ operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
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
from flag_gems.experimental_ops.copy_ import copy_ as gems_copy_

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark


@pytest.mark.copy_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_copy__default(shape, dtype, non_blocking):
    dst_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_dst = dst_base.clone()
    ref_src = src_base.clone()
    ref_out = torch.ops.aten.copy_(ref_dst, ref_src, non_blocking)

    act_dst = dst_base.clone()
    act_src = src_base.clone()
    with flag_gems.use_gems():
        act_out = gems_copy_(act_dst, act_src, non_blocking)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.copy_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_copy__tensor_overload(shape, dtype):
    dst_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src_base = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_dst = dst_base.clone()
    ref_src = src_base.clone()
    ref_out = torch.ops.aten.copy_.Tensor(ref_dst, ref_src)

    act_dst = dst_base.clone()
    act_src = src_base.clone()
    with flag_gems.use_gems():
        act_out = torch.ops.aten.copy_.Tensor(act_dst, act_src)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.copy_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0, 1, -3, 42])
def test_copy__scalar_int(dtype, value):
    dst_base = torch.zeros((), dtype=dtype, device=flag_gems.device)

    ref_dst = dst_base.clone()
    ref_out = torch.ops.aten.copy_.int(ref_dst, value)

    act_dst = dst_base.clone()
    with flag_gems.use_gems():
        act_out = torch.ops.aten.copy_.int(act_dst, value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.copy_
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, -1.5, 3.14, 100.5])
def test_copy__scalar_float(dtype, value):
    dst_base = torch.zeros((), dtype=dtype, device=flag_gems.device)

    ref_dst = dst_base.clone()
    ref_out = torch.ops.aten.copy_.float(ref_dst, value)

    act_dst = dst_base.clone()
    with flag_gems.use_gems():
        act_out = torch.ops.aten.copy_.float(act_dst, value)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.copy_
def test_perf_aten_copy_():
    # Define input generation logic matching the operator arguments
    def copy__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=copy__input_fn,
        op_name="copy_",
        torch_op=torch.ops.aten.copy_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
