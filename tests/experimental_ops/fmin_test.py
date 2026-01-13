# FMIN operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.fmin import fmin as gems_fmin
from flag_gems.experimental_ops.fmin import fmin_out as gems_fmin_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

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


@pytest.mark.fmin
@pytest.mark.parametrize(
    "shape_self,shape_other",
    [
        ((2, 3), (2, 3)),
        ((128, 256), (128, 256)),
        ((512, 512), (512, 512)),
        ((4, 1, 32), (1, 32)),
        ((1, 5, 7), (5, 1, 7)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fmin_tensor(shape_self, shape_other, dtype):
    self_tensor = torch.randn(shape_self, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape_other, dtype=dtype, device=flag_gems.device)

    self_mask = torch.rand(shape_self, device=flag_gems.device) < 0.05
    other_mask = torch.rand(shape_other, device=flag_gems.device) < 0.05
    self_tensor[self_mask] = float("nan")
    other_tensor[other_mask] = float("nan")

    ref_self = to_reference(self_tensor)
    ref_other = to_reference(other_tensor)
    ref_out = torch.ops.aten.fmin(ref_self, ref_other)

    with flag_gems.use_gems():
        act_out = gems_fmin(self_tensor, other_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=True)


@pytest.mark.fmin
@pytest.mark.parametrize(
    "shape_self,shape_other",
    [
        ((2, 3), (2, 3)),
        ((128, 256), (128, 256)),
        ((512, 512), (512, 512)),
        ((4, 1, 32), (1, 32)),
        ((1, 5, 7), (5, 1, 7)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fmin_out(shape_self, shape_other, dtype):
    self_tensor = torch.randn(shape_self, dtype=dtype, device=flag_gems.device)
    other_tensor = torch.randn(shape_other, dtype=dtype, device=flag_gems.device)

    self_mask = torch.rand(shape_self, device=flag_gems.device) < 0.05
    other_mask = torch.rand(shape_other, device=flag_gems.device) < 0.05
    self_tensor[self_mask] = float("nan")
    other_tensor[other_mask] = float("nan")

    ref_self = to_reference(self_tensor)
    ref_other = to_reference(other_tensor)
    ref_broadcast_self, ref_broadcast_other = torch.broadcast_tensors(
        ref_self, ref_other
    )
    ref_out_buf = torch.empty(
        ref_broadcast_self.shape,
        dtype=ref_broadcast_self.dtype,
        device=ref_broadcast_self.device,
    )
    ref_out = torch.ops.aten.fmin.out(ref_self, ref_other, out=ref_out_buf)

    with flag_gems.use_gems():
        act_broadcast_self, act_broadcast_other = torch.broadcast_tensors(
            self_tensor, other_tensor
        )
        act_out_buf = torch.empty(
            act_broadcast_self.shape, dtype=dtype, device=flag_gems.device
        )
        act_out = gems_fmin_out(self_tensor, other_tensor, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype, equal_nan=True)


@pytest.mark.fmin
def test_perf_aten_fmin():
    # Define input generation logic matching the operator arguments
    def fmin_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=fmin_input_fn,
        op_name="fmin",
        torch_op=torch.ops.aten.fmin,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
