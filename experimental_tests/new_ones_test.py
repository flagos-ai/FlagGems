# NEW_ONES operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.new_ones import new_ones as gems_new_ones  # noqa: E402
from flag_gems.experimental_ops.new_ones import (  # noqa: E402
    new_ones_out as gems_new_ones_out,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.new_ones
@pytest.mark.parametrize("self_shape", [(2, 3), (128, 256)])
@pytest.mark.parametrize("size", [(2, 3), (128, 256), (32, 16, 8), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_new_ones_default(self_shape, size, dtype):
    self_tensor = torch.randn(self_shape, dtype=torch.float32, device=flag_gems.device)

    ref_self = to_reference(self_tensor)
    ref_out = torch.ops.aten.new_ones(ref_self, size, dtype=dtype)

    with flag_gems.use_gems():
        act_out = gems_new_ones(self_tensor, size, dtype=dtype)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.new_ones
@pytest.mark.parametrize("self_shape", [(2, 3), (64, 64)])
@pytest.mark.parametrize("size", [(2, 3), (128, 256), (16, 8, 4), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_new_ones_out(self_shape, size, dtype):
    self_tensor = torch.randn(self_shape, dtype=torch.float32, device=flag_gems.device)

    ref_self = to_reference(self_tensor)
    ref_out_buf = torch.empty(size, device=ref_self.device, dtype=dtype)
    ref_out = torch.ops.aten.new_ones.out(ref_self, size, out=ref_out_buf)

    with flag_gems.use_gems():
        act_out_buf = torch.empty(size, device=flag_gems.device, dtype=dtype)
        act_out = gems_new_ones_out(self_tensor, size, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.new_ones
def test_perf_aten_new_ones():
    # Define input generation logic matching the operator arguments
    def new_ones_input_fn(shape, dtype, device):
        inp1 = torch.randn(
            shape, dtype=torch.float32, device=flag_gems.device
        )  # self_tensor
        # yield inputs as required by the operator (size as position,
        # dtype as keyword)
        yield inp1, shape

    # Create a wrapper function to handle dtype as keyword argument
    def new_ones_wrapper(self_tensor, size):
        return torch.ops.aten.new_ones(self_tensor, size, dtype=self_tensor.dtype)

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=new_ones_input_fn,
        op_name="new_ones",
        torch_op=new_ones_wrapper,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
