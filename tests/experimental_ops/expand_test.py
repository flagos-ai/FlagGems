# EXPAND operator test

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


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.expand import expand as gems_expand  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402


@pytest.mark.expand
@pytest.mark.parametrize(
    "in_shape_out",
    [
        ((2, 3), (2, 3)),
        ((1, 3), (5, 3)),
        ((2, 1, 4), (2, 7, 4)),
        ((128, 1), (128, 256)),
        ((64, 1, 32), (64, 512, 32)),
        ((2, 3), (-1, 3)),
        ((1, 3), (-1, 3)),
        ((1, 1), (128, 256)),
        ((16, 1, 1, 8), (16, 32, 64, 8)),
        ((32, 4), (32, -1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implicit", [False, True])
def test_expand_tensor(in_shape_out, dtype, implicit):
    in_shape, out_size = in_shape_out
    input_tensor = torch.randn(in_shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.expand(ref_input, out_size, implicit=implicit)

    with flag_gems.use_gems():
        act_out = gems_expand(input_tensor, out_size, implicit=implicit)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.expand
@pytest.mark.parametrize(
    "base_shape,op,out_size",
    [
        ((16, 1, 8), "transpose", (8, 32, 16)),
        ((4, 1, 5, 1), "permute", (5, 4, 7, 9)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implicit", [False, True])
def test_expand_noncontiguous(base_shape, op, out_size, dtype, implicit):
    base = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_base = base.clone()

    if op == "transpose":
        input_tensor = base.transpose(0, 2)
        ref_input = ref_base.transpose(0, 2)
    else:
        input_tensor = base.permute(2, 0, 3, 1)
        ref_input = ref_base.permute(2, 0, 3, 1)

    ref_out = torch.ops.aten.expand(ref_input, out_size, implicit=implicit)

    with flag_gems.use_gems():
        act_out = gems_expand(input_tensor, out_size, implicit=implicit)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.expand
def test_perf_aten_expand():
    # Define input generation logic matching the operator arguments
    def expand_input_fn(shape, dtype, device):
        # Generate input tensor and output size
        input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # Create output size based on the input shape
        out_size = tuple(
            s if s != -1 else input_tensor.size(i) for i, s in enumerate(shape)
        )
        yield input_tensor, out_size

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=expand_input_fn,
        op_name="expand",
        torch_op=torch.ops.aten.expand,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
