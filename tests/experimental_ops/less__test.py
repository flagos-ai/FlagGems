# LESS_ operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.less_ import less__Scalar, less__Tensor  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
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


@pytest.mark.less_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("other", [-0.5, 0.0, 1.25])
def test_less__scalar(shape, dtype, other):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_out = torch.ops.aten.less_(ref_inp, other)

    with flag_gems.use_gems():
        act_inp = inp.clone()
        act_out = less__Scalar(act_inp, other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.less_
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_less__tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp)
    ref_other = to_reference(other)
    ref_out = torch.ops.aten.less_(ref_inp, ref_other)

    with flag_gems.use_gems():
        act_inp = inp.clone()
        act_other = other.clone()
        act_out = less__Tensor(act_inp, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.less_
def test_perf_aten_less_():
    # Define input generation logic matching the operator arguments
    def less__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark - using less__Tensor for performance test
    def less__Tensor_wrapper(self, other):
        return less__Tensor(self, other)

    bench = GenericBenchmark(
        input_fn=less__input_fn,
        op_name="less_",
        torch_op=torch.ops.aten.less_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    # Replace the op with less__Tensor
    bench.op = less__Tensor_wrapper

    return bench.run()
