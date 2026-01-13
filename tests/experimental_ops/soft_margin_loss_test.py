# SOFT_MARGIN_LOSS operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.soft_margin_loss import (
    soft_margin_loss as gems_soft_margin_loss,
)
from flag_gems.experimental_ops.soft_margin_loss import (
    soft_margin_loss_out as gems_soft_margin_loss_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from benchmark.performance_utils import GenericBenchmark
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


@pytest.mark.soft_margin_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss_tensor(shape, dtype, reduction):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (torch.randint(0, 2, shape, device=flag_gems.device).to(dtype) * 2) - 1

    ref_self = to_reference(self)
    ref_target = to_reference(target)
    ref_out = torch.ops.aten.soft_margin_loss(ref_self, ref_target, reduction)

    with flag_gems.use_gems():
        act_out = gems_soft_margin_loss(self, target, reduction)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.soft_margin_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss_out(shape, dtype, reduction):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (torch.randint(0, 2, shape, device=flag_gems.device).to(dtype) * 2) - 1

    ref_self = to_reference(self)
    ref_target = to_reference(target)

    if reduction == 0:
        out_ref = torch.empty(shape, dtype=dtype, device=ref_self.device)
        out_act = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        out_ref = torch.empty((), dtype=dtype, device=ref_self.device)
        out_act = torch.empty((), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.soft_margin_loss.out(
        ref_self, ref_target, reduction, out=out_ref
    )

    with flag_gems.use_gems():
        act_out = gems_soft_margin_loss_out(self, target, reduction, out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.soft_margin_loss
def test_perf_aten_soft_margin_loss():
    # Define input generation logic matching the operator arguments
    def soft_margin_loss_input_fn(shape, dtype, device):
        self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        target = (torch.randint(0, 2, shape, device=flag_gems.device).to(dtype) * 2) - 1
        yield self, target

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=soft_margin_loss_input_fn,
        op_name="soft_margin_loss",
        torch_op=torch.ops.aten.soft_margin_loss,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
